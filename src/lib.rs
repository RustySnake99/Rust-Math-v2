use core::f64;
use pyo3::prelude::*;

#[pyfunction]
fn cosine_similarity(v1: Vec<f64>, v2: Vec<f64>) -> PyResult<f64> {
    if v1.len() != v2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input vectors must have the same length!"));
    }
    let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let normal_v1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let normal_v2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

    Ok(dot_product / (normal_v1 * normal_v2))
}

#[pyfunction]
fn dot_product(v1: Vec<f64>, v2: Vec<f64>) -> PyResult<f64> {
    if v1.len() != v2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input vectors must have the same length!"));
    }
    Ok(v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum())
}

#[pyfunction]
fn cross_product(v1: Vec<f64>, v2: Vec<f64>) -> PyResult<Vec<f64>> {
    if v1.len() != 3 || v2.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input vectors must be 3-dimensional!"));
    }
    Ok(vec![
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ])
}

#[pyfunction]
fn vector_magnitude(v: Vec<f64>) -> PyResult<f64> {
    Ok(v.iter().map(|x| x * x).sum::<f64>().sqrt())
}

#[pyfunction]
fn unit_vector(v: Vec<f64>) -> PyResult<Vec<f64>> {
    let magnitude = vector_magnitude(v.clone())?;
    if magnitude == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot compute unit vector of a zero vector!"));
    }
    Ok(v.iter().map(|x| x / magnitude).collect())
}

#[pyfunction]
fn angle_between(v1: Vec<f64>, v2: Vec<f64>) -> PyResult<f64> {
    let cosine_similarity = cosine_similarity(v1, v2)?;
    Ok(cosine_similarity.acos())
}

#[pyfunction]
fn matrix_product(m1: Vec<Vec<f64>>, m2: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if m1[0].len() != m2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Number of columns in the first matrix must equal the number of rows in the second matrix!"));
    }
    let (result_rows, result_cols) = (m1.len(), m2[0].len());
    let mut result = vec![vec![0.0; result_cols]; result_rows];

    for i in 0..result_rows {
        for j in 0..result_cols {
            result[i][j] = (0..m2.len()).map(|k| m1[i][k] * m2[k][j]).sum();
        }
    }
    Ok(result)
}

#[pyfunction]
fn mean(v: Vec<f64>) -> PyResult<f64> {
    if v.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot compute mean of an empty vector!"));
    }
    Ok(v.iter().sum::<f64>() / v.len() as f64)
}

#[pyfunction]
fn variance(v: Vec<f64>) -> PyResult<f64> {
    if v.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("At least two elements are required to compute variance!"));
    }
    let mean_value = mean(v.clone())?;
    Ok(v.iter().map(|x| (x - mean_value).powi(2)).sum::<f64>() / v.len() as f64)
}

#[pyfunction]
fn standard_deviation(v: Vec<f64>) -> PyResult<f64> {
    Ok(variance(v)? .sqrt())
}

#[pyfunction]
fn softmax(v: Vec<f64>) -> PyResult<Vec<f64>> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    Ok(exp_values.iter().map(|x| x / exp_values.iter().sum::<f64>()).collect())
}

#[pymodule]
fn RustMath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(cross_product, m)?)?;
    m.add_function(wrap_pyfunction!(vector_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(unit_vector, m)?)?;
    m.add_function(wrap_pyfunction!(angle_between, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_product, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(standard_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    Ok(())
}