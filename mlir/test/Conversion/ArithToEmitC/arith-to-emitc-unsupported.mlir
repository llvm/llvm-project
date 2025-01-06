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
func.func @arith_cast_f80(%arg0: f80) -> i32 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : f80 to i32
  return %t: i32
}

// -----

func.func @arith_cast_f128(%arg0: f128) -> i32 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : f128 to i32
  return %t: i32
}


// -----

func.func @arith_cast_to_f80(%arg0: i32) -> f80 {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %t = arith.sitofp %arg0 : i32 to f80
  return %t: f80
}

// -----

func.func @arith_cast_to_f128(%arg0: i32) -> f128 {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %t = arith.sitofp %arg0 : i32 to f128
  return %t: f128
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

func.func @arith_cmpf_vector(%arg0: vector<5xf32>, %arg1: vector<5xf32>) -> vector<5xi1> {
  // expected-error @+1 {{failed to legalize operation 'arith.cmpf'}}
  %t = arith.cmpf uno, %arg0, %arg1 : vector<5xf32>
  return %t: vector<5xi1>
}

// -----

func.func @arith_cmpf_tensor(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xi1> {
  // expected-error @+1 {{failed to legalize operation 'arith.cmpf'}}
  %t = arith.cmpf uno, %arg0, %arg1 : tensor<5xf32>
  return %t: tensor<5xi1>
}

// -----

func.func @arith_negf_f80(%arg0: f80) -> f80 {
  // expected-error @+1 {{failed to legalize operation 'arith.negf'}}
  %n = arith.negf %arg0 : f80
  return %n: f80
}

// -----

func.func @arith_negf_tensor(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.negf'}}
  %n = arith.negf %arg0 : tensor<5xf32>
  return %n: tensor<5xf32>
}

// -----

func.func @arith_negf_vector(%arg0: vector<5xf32>) -> vector<5xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.negf'}}
  %n = arith.negf %arg0 : vector<5xf32>
  return %n: vector<5xf32>
}

// -----

func.func @arith_extsi_i1_to_i32(%arg0: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.extsi'}}
  %idx = arith.extsi %arg0 : i1 to i32
  return
}

// -----

func.func @arith_shli_i1(%arg0: i1, %arg1: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.shli'}}
  %shli = arith.shli %arg0, %arg1 : i1
  return
}

// -----

func.func @arith_shrsi_i1(%arg0: i1, %arg1: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.shrsi'}}
  %shrsi = arith.shrsi %arg0, %arg1 : i1
  return
}

// -----

func.func @arith_shrui_i1(%arg0: i1, %arg1: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.shrui'}}
  %shrui = arith.shrui %arg0, %arg1 : i1
  return
}

// -----

func.func @arith_divui_vector(%arg0: vector<5xi32>, %arg1: vector<5xi32>) -> vector<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.divui'}}
  %divui = arith.divui %arg0, %arg1 : vector<5xi32>
  return %divui: vector<5xi32>
}

// -----

func.func @arith_remui_vector(%arg0: vector<5xi32>, %arg1: vector<5xi32>) -> vector<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.remui'}}
  %divui = arith.remui %arg0, %arg1 : vector<5xi32>
  return %divui: vector<5xi32>
}
