// RUN: mlir-opt %s --test-next-access --split-input-file | FileCheck %s

// CHECK-LABEL: @trivial
func.func @trivial(%arg0: memref<f32>, %arg1: f32) -> f32 {
  // CHECK:      name = "store"
  // CHECK-SAME: next_access = ["unknown", ["load"]]
  memref.store %arg1, %arg0[] {name = "store"} : memref<f32>
  // CHECK:      name = "load"
  // CHECK-SAME: next_access = ["unknown"]
  %0 = memref.load %arg0[] {name = "load"} : memref<f32>
  return %0 : f32
}

// CHECK-LABEL: @chain
func.func @chain(%arg0: memref<f32>, %arg1: f32) -> f32 {
  // CHECK:      name = "store"
  // CHECK-SAME: next_access = ["unknown", ["load 1"]]
  memref.store %arg1, %arg0[] {name = "store"} : memref<f32>
  // CHECK:      name = "load 1"
  // CHECK-SAME: next_access = {{\[}}["load 2"]]
  %0 = memref.load %arg0[] {name = "load 1"} : memref<f32>
  // CHECK:      name = "load 2"
  // CHECK-SAME: next_access = ["unknown"]
  %1 = memref.load %arg0[] {name = "load 2"} : memref<f32>
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}

// CHECK-LABEL: @branch
func.func @branch(%arg0: memref<f32>, %arg1: f32, %arg2: i1) -> f32 {
  // CHECK:      name = "store"
  // CHECK-SAME: next_access = ["unknown", ["load 1", "load 2"]]
  memref.store %arg1, %arg0[] {name = "store"} : memref<f32>
  cf.cond_br %arg2, ^bb0, ^bb1

^bb0:
  %0 = memref.load %arg0[] {name = "load 1"} : memref<f32>
  cf.br ^bb2(%0 : f32)

^bb1:
  %1 = memref.load %arg0[] {name = "load 2"} : memref<f32>
  cf.br ^bb2(%1 : f32)

^bb2(%phi: f32):
  return %phi : f32
}

// CHECK-LABEL @dead_branch
func.func @dead_branch(%arg0: memref<f32>, %arg1: f32) -> f32 {
  // CHECK:      name = "store"
  // CHECK-SAME: next_access = ["unknown", ["load 2"]]
  memref.store %arg1, %arg0[] {name = "store"} : memref<f32>
  cf.br ^bb1

^bb0:
  // CHECK:      name = "load 1"
  // CHECK-SAME: next_access = "not computed"
  %0 = memref.load %arg0[] {name = "load 1"} : memref<f32>
  cf.br ^bb2(%0 : f32)

^bb1:
  %1 = memref.load %arg0[] {name = "load 2"} : memref<f32>
  cf.br ^bb2(%1 : f32)

^bb2(%phi: f32):
  return %phi : f32
}

// CHECK-LABEL: @loop
func.func @loop(%arg0: memref<?xf32>, %arg1: f32, %arg2: index, %arg3: index, %arg4: index) -> f32 {
  %c0 = arith.constant 0.0 : f32
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["loop"], "unknown"]
  // The above is not entirely correct when the loop has 0 iterations, but 
  // the region control flow specificaiton is currently incapable of
  // specifying that.
  memref.load %arg0[%arg4] {name = "pre"} : memref<?xf32>
  %l = scf.for %i = %arg2 to %arg3 step %arg4 iter_args(%ia = %c0) -> (f32) {
    // CHECK:      name = "loop"
    // CHECK-SAME: next_access = {{\[}}["outside", "loop"], "unknown"]
    %0 = memref.load %arg0[%i] {name = "loop"} : memref<?xf32>
    %1 = arith.addf %ia, %0 : f32
    scf.yield %1 : f32
  }
  %v = memref.load %arg0[%arg3] {name = "outside"} : memref<?xf32>
  %2 = arith.addf %v, %l : f32
  return %2 : f32
}

// CHECK-LABEL: @conditional
func.func @conditional(%cond: i1, %arg0: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["then"]]
  // The above is not entirely correct when the condition is false, but 
  // the region control flow specificaiton is currently incapable of
  // specifying that.
  memref.load %arg0[] {name = "pre"}: memref<f32>
  scf.if %cond {
    // CHECK:      name = "then"
    // CHECK-SAME: next_access = {{\[}}["post"]]
    memref.load %arg0[] {name = "then"} : memref<f32>
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// CHECK-LABEL: @two_sided_conditional
func.func @two_sided_conditional(%cond: i1, %arg0: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["then", "else"]]
  memref.load %arg0[] {name = "pre"}: memref<f32>
  scf.if %cond {
    // CHECK:      name = "then"
    // CHECK-SAME: next_access = {{\[}}["post"]]
    memref.load %arg0[] {name = "then"} : memref<f32>
  } else {
    // CHECK:      name = "else"
    // CHECK-SAME: next_access = {{\[}}["post"]]
    memref.load %arg0[] {name = "else"} : memref<f32>
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// CHECK-LABEL: @dead_conditional
func.func @dead_conditional(%arg0: memref<f32>) {
  %false = arith.constant 0 : i1
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = ["unknown"]
  // The above is not entirely correct when the condition is false, but 
  // the region control flow specificaiton is currently incapable of
  // specifying that.
  memref.load %arg0[] {name = "pre"}: memref<f32>
  scf.if %false {
    // CHECK:      name = "then"
    // CHECK-SAME: next_access = "not computed"
    memref.load %arg0[] {name = "then"} : memref<f32>
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// CHECK-LABEL: @known_conditional
func.func @known_conditional(%arg0: memref<f32>) {
  %false = arith.constant 0 : i1
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["else"]]
  memref.load %arg0[] {name = "pre"}: memref<f32>
  scf.if %false {
    // CHECK:      name = "then"
    // CHECK-SAME: next_access = "not computed"
    memref.load %arg0[] {name = "then"} : memref<f32>
  } else {
    // CHECK:      name = "else"
    // CHECK-SAME: next_access = {{\[}}["post"]]
    memref.load %arg0[] {name = "else"} : memref<f32>
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// CHECK-LABEL: @loop_cf
func.func @loop_cf(%arg0: memref<?xf32>, %arg1: f32, %arg2: index, %arg3: index, %arg4: index) -> f32 {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["loop", "outside"], "unknown"]
  %0 = memref.load %arg0[%arg4] {name = "pre"} : memref<?xf32>
  cf.br ^bb1(%arg2, %cst : index, f32)
^bb1(%1: index, %2: f32):
  %3 = arith.cmpi slt, %1, %arg3 : index
  cf.cond_br %3, ^bb2, ^bb3
^bb2:
  // CHECK:      name = "loop"
  // CHECK-SAME: next_access = {{\[}}["loop", "outside"], "unknown"]
  %4 = memref.load %arg0[%1] {name = "loop"} : memref<?xf32>
  %5 = arith.addf %2, %4 : f32
  %6 = arith.addi %1, %arg4 : index
  cf.br ^bb1(%6, %5 : index, f32)
^bb3:
  %7 = memref.load %arg0[%arg3] {name = "outside"} : memref<?xf32>
  %8 = arith.addf %7, %2 : f32
  return %8 : f32
}

// CHECK-LABEL @conditional_cf
func.func @conditional_cf(%arg0: i1, %arg1: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["then", "post"]]
  %0 = memref.load %arg1[] {name = "pre"} : memref<f32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK:      name = "then"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  %1 = memref.load %arg1[] {name = "then"} : memref<f32>
  cf.br ^bb2
^bb2:
  %2 = memref.load %arg1[] {name = "post"} : memref<f32>
  return
}

// CHECK-LABEL: @two_sided_conditional_cf
func.func @two_sided_conditional_cf(%arg0: i1, %arg1: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["then", "else"]]
  %0 = memref.load %arg1[] {name = "pre"} : memref<f32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK:      name = "then"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  %1 = memref.load %arg1[] {name = "then"} : memref<f32>
  cf.br ^bb3
^bb2:
  // CHECK:      name = "else"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  %2 = memref.load %arg1[] {name = "else"} : memref<f32>
  cf.br ^bb3
^bb3:
  %3 = memref.load %arg1[] {name = "post"} : memref<f32>
  return
}

// CHECK-LABEL: @dead_conditional_cf
func.func @dead_conditional_cf(%arg0: memref<f32>) {
  %false = arith.constant false
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  %0 = memref.load %arg0[] {name = "pre"} : memref<f32>
  cf.cond_br %false, ^bb1, ^bb2
^bb1:
  // CHECK:      name = "then"
  // CHECK-SAME: next_access = "not computed"
  %1 = memref.load %arg0[] {name = "then"} : memref<f32>
  cf.br ^bb2
^bb2:
  %2 = memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// CHECK-LABEL: @known_conditional_cf
func.func @known_conditional_cf(%arg0: memref<f32>) {
  %false = arith.constant false
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["else"]]
  %0 = memref.load %arg0[] {name = "pre"} : memref<f32>
  cf.cond_br %false, ^bb1, ^bb2
^bb1:
  // CHECK:      name = "then"
  // CHECK-SAME: next_access = "not computed"
  %1 = memref.load %arg0[] {name = "then"} : memref<f32>
  cf.br ^bb3
^bb2:
  // CHECK:      name = "else"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  %2 = memref.load %arg0[] {name = "else"} : memref<f32>
  cf.br ^bb3
^bb3:
  %3 = memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

func.func private @callee1(%arg0: memref<f32>) {
  // CHECK:      name = "callee1"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  memref.load %arg0[] {name = "callee1"} : memref<f32>
  return
}

func.func private @callee2(%arg0: memref<f32>) {
  // CHECK:      name = "callee2"
  // CHECK-SAME: next_access = "not computed"
  memref.load %arg0[] {name = "callee2"} : memref<f32>
  return
}

// CHECK-LABEL: @simple_call
func.func @simple_call(%arg0: memref<f32>) {
  // CHECK:      name = "caller"
  // CHECK-SAME: next_access = {{\[}}["callee1"]]
  memref.load %arg0[] {name = "caller"} : memref<f32>
  func.call @callee1(%arg0) : (memref<f32>) -> ()
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: @infinite_recursive_call
func.func @infinite_recursive_call(%arg0: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["pre"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  func.call @infinite_recursive_call(%arg0) : (memref<f32>) -> ()
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: @recursive_call
func.func @recursive_call(%arg0: memref<f32>, %cond: i1) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["pre"]]
  // The above is not entirely correct when the condition is false, but 
  // the region control flow specificaiton is currently incapable of
  // specifying that.
  memref.load %arg0[] {name = "pre"} : memref<f32>
  scf.if %cond {
    func.call @recursive_call(%arg0, %cond) : (memref<f32>, i1) -> ()
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: @recursive_call_cf
func.func @recursive_call_cf(%arg0: memref<f32>, %cond: i1) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["pre", "post"]]
  %0 = memref.load %arg0[] {name = "pre"} : memref<f32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  call @recursive_call_cf(%arg0, %cond) : (memref<f32>, i1) -> ()
  cf.br ^bb2
^bb2:
  %2 = memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

func.func private @callee1(%arg0: memref<f32>) {
  // CHECK:      name = "callee1"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  memref.load %arg0[] {name = "callee1"} : memref<f32>
  return
}

func.func private @callee2(%arg0: memref<f32>) {
  // CHECK:      name = "callee2"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  memref.load %arg0[] {name = "callee2"} : memref<f32>
  return
}

func.func @conditonal_call(%arg0: memref<f32>, %cond: i1) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["callee1", "callee2"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  scf.if %cond {
    func.call @callee1(%arg0) : (memref<f32>) -> ()
  } else {
    func.call @callee2(%arg0) : (memref<f32>) -> ()
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}
