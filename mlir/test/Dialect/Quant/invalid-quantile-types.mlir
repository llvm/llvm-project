// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Verify errors (caught by verify(), reached through getChecked())
//===----------------------------------------------------------------------===//

// Storage type must be an integer or float.
// expected-error @+1 {{storage type must be an integer or float type}}
func.func private @invalid_storage_type() -> !quant.quantile<tensor<1xf32>:f32, {1.0}>

// -----

// Quantile (expressed) type must be a float.
// expected-error @+1 {{quantile type must be a float type}}
func.func private @invalid_quantile_type() -> !quant.quantile<ui4:i8, {1.0, 0.0, -1.0}>

// -----

// Quantile LUT must not be empty.
// expected-error @+1 {{quantile values must not be empty}}
func.func private @empty_quantiles() -> !quant.quantile<ui4:f16, {}>

// -----

// LUT size must match the number of representable storage values.
// ui4 has 16 representable values [0,15], but only 3 are provided.
// expected-error @+1 {{quantile LUT size (3) must equal the number of representable storage values (16)}}
func.func private @wrong_lut_size() -> !quant.quantile<ui4:f16, {-1.0,0.0,1.0}>

// -----

// Explicit storage range: min must be strictly less than max.
// si4 default range is [-8,7]; explicit 5:3 has min > max.
// expected-error @+1 {{storage min must be less than storage max}}
func.func private @invalid_range_order() -> !quant.quantile<si4:f32, {-2.0,-1.875,-1.75,-1.625,-1.5,-1.375,-1.25,-1.125,-1.0,-0.875,-0.75,-0.625,-0.5,-0.375,-0.25,-0.125}, <5:3>>

// -----

// LUT size must match the total representable values of the storage type.
// f4E2M1FN has 16 representable values regardless of explicit range, but only 3 are provided.
// expected-error @+1 {{quantile LUT size (3) must equal the number of representable storage values (13)}}
func.func private @wrong_lut_size_with_range() -> !quant.quantile<f4E2M1FN:f16, {-1.0,0.0,1.0}, <-6:6>>
