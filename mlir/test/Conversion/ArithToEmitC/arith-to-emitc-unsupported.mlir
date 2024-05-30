// RUN: mlir-opt -split-input-file -convert-arith-to-emitc -verify-diagnostics %s

func.func @arith_cast_tensor(%arg0: tensor<5xf32>) -> tensor<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : tensor<5xf32> to tensor<5xi32>
  return %t: tensor<5xi32>
}

// -----

func.func @arith_cast_vector(%arg0: vector<5xf32>) -> vector<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : vector<5xf32> to vector<5xi32>
  return %t: vector<5xi32>
}

// -----

func.func @arith_cast_bf16(%arg0: bf16) -> i32 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : bf16 to i32
  return %t: i32
}

// -----

func.func @arith_cast_f16(%arg0: f16) -> i32 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : f16 to i32
  return %t: i32
}


// -----

func.func @arith_cast_to_bf16(%arg0: i32) -> bf16 {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %t = arith.sitofp %arg0 : i32 to bf16
  return %t: bf16
}

// -----

func.func @arith_cast_to_f16(%arg0: i32) -> f16 {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %t = arith.sitofp %arg0 : i32 to f16
  return %t: f16
}

// -----

func.func @arith_cast_fptosi_i1(%arg0: f32) -> i1 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : f32 to i1
  return %t: i1
}

// -----

func.func @arith_cast_fptoui_i1(%arg0: f32) -> i1 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptoui'}}
  %t = arith.fptoui %arg0 : f32 to i1
  return %t: i1
}

// -----

func.func @arith_extsi_i1_to_i32(%arg0: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.extsi'}}
  %idx = arith.extsi %arg0 : i1 to i32
  return
}
