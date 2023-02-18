// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @bad_source_types(%a: vector<2xf32>, %b: vector<4xf16>,
                                %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected both non-f8 source operand types to match exactly}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 1 : i32, blocks = 2 : i32,
    abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<2xf32>, vector<4xf16>, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @bad_source_types_f8(%a: vector<8xf8E5M2FNUZ>, %b: vector<8xi8>,
                                %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected both source operands to have f8 elements}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 1 : i32, blocks = 2 : i32,
    abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<8xf8E5M2FNUZ>, vector<8xi8>, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @bad_source_arguments(%a: vector<2xf32>, %b: vector<2xf32>,
                                %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected 1 source values for this operation but got 2}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 1 : i32, blocks = 2 : i32,
    abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<2xf32>, vector<2xf32>, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @bad_source_arguments_i8(%a: vector<8xi8>, %b: vector<8xi8>,
                                   %c: vector<4xi32>) -> vector<4xi32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected 4 source values for this operation but got 8}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 4 : i32, blocks = 2 : i32,
    abid = 0 : i32, cbsz = 0 : i32} blgp = none : vector<8xi8>, vector<8xi8>, vector<4xi32>
  func.return %d : vector<4xi32>
}

// -----

func.func @bad_dest_type(%a: f32, %b: f32, %c: vector<16xf32>) -> vector<16xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op expected 32 result values for this operation but got 16}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 1 : i32, blocks = 2 : i32,
    abid = 0 : i32, cbsz = 0 : i32} blgp = none : f32, f32, vector<16xf32>
  return %d : vector<16xf32>
}

// -----

func.func @f64_permuting_b(%a: f64, %b: f64, %c: vector<4xf64>) -> vector<4xf64> {
  // expected-error@+1 {{'amdgpu.mfma' op double-precision ops do not support permuting lanes of B}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 16 : i32, n = 16 : i32, k = 4 : i32, blocks = 1 : i32,
    abid = 0 : i32, cbsz = 0 : i32} blgp = bcast_first_32 : f64, f64, vector<4xf64>
  return %d : vector<4xf64>
}

// -----

func.func @f64_permuting_a(%a: f64, %b: f64, %c: vector<4xf64>) -> vector<4xf64> {
  // expected-error@+1 {{'amdgpu.mfma' op double-precision ops do not support permuting lanes of A}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 16 : i32, n = 16 : i32, k = 4 : i32, blocks = 1 : i32,
    abid = 0 : i32, cbsz = 1 : i32} blgp = none : f64, f64, vector<4xf64>
  return %d : vector<4xf64>
}

// -----

func.func @abid_without_bradcast(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op block ID for permuting A (abid) must be below 2 ** cbsz}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 1 : i32, blocks = 2 : i32,
    abid = 1 : i32, cbsz = 0 : i32} blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @abid_too_large(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op block ID for permuting A (abid) must be below 2 ** cbsz}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 1 : i32, blocks = 2 : i32,
    abid = 2 : i32, cbsz = 1 : i32} blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}

// -----

func.func @no_negation(%a: f32, %b: f32, %c: vector<32xf32>) -> vector<32xf32> {
  // expected-error@+1 {{'amdgpu.mfma' op negation flags only available for double-precision operations}}
  %d = amdgpu.mfma %a * %b + %c {
    m = 32 : i32, n = 32 : i32, k = 1 : i32, blocks = 2 : i32,
    abid = 0 : i32, cbsz = 0 : i32, negateA} blgp = none : f32, f32, vector<32xf32>
  func.return %d : vector<32xf32>
}
