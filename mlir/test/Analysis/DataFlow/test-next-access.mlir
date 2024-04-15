// RUN: mlir-opt %s --test-next-access --split-input-file |\
// RUN:             FileCheck %s --check-prefixes=CHECK,IP
// RUN: mlir-opt %s --test-next-access='interprocedural=false' \
// RUN:             --split-input-file |\
// RUN:             FileCheck %s --check-prefixes=CHECK,LOCAL
// RUN: mlir-opt %s --test-next-access='assume-func-reads=true' \
// RUN:             --split-input-file |\
// RUN:             FileCheck %s --check-prefixes=CHECK,IP_AR
// RUN: mlir-opt %s \
// RUN:      --test-next-access='interprocedural=false assume-func-reads=true' \
// RUN:      --split-input-file | FileCheck %s --check-prefixes=CHECK,LC_AR

// Check prefixes are as follows:
// 'check': common for all runs;
// 'ip_ar': interpocedural runs assuming calls to external functions read
//          all arguments;
// 'ip': interprocedural runs not assuming function calls reading;
// 'local': local (non-interprocedural) analysis not assuming calls reading;
// 'lc_ar': local analysis assuming external calls reading all arguments.

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
  // CHECK-SAME: next_access = {{\[}}["outside", "loop"], "unknown"]
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
  // CHECK-SAME: next_access = {{\[}}["post", "then"]]
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
  // CHECK-SAME: next_access = {{\[}}["post"]]
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
  // IP:         name = "callee1"
  // IP-SAME:    next_access = {{\[}}["post"]]
  // LOCAL:      name = "callee1"
  // LOCAL-SAME: next_access = ["unknown"]
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
  // IP:         name = "caller"
  // IP-SAME:    next_access = {{\[}}["callee1"]]
  // LOCAL:      name = "caller"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "caller"
  // LC_AR-SAME: next_access = {{\[}}["call"]]
  memref.load %arg0[] {name = "caller"} : memref<f32>
  func.call @callee1(%arg0) {name = "call"} : (memref<f32>) -> ()
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: @infinite_recursive_call
func.func @infinite_recursive_call(%arg0: memref<f32>) {
  // IP:         name = "pre"
  // IP-SAME:    next_access = {{\[}}["pre"]]
  // LOCAL:      name = "pre"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "pre"
  // LC_AR-SAME: next_access = {{\[}}["call"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  func.call @infinite_recursive_call(%arg0) {name = "call"} : (memref<f32>) -> ()
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: @recursive_call
func.func @recursive_call(%arg0: memref<f32>, %cond: i1) {
  // IP:         name = "pre"
  // IP-SAME:    next_access = {{\[}}["post", "pre"]]
  // LOCAL:      name = "pre"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "pre"
  // LC_AR-SAME: next_access = {{\[}}["post", "call"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  scf.if %cond {
    func.call @recursive_call(%arg0, %cond) {name = "call"} : (memref<f32>, i1) -> ()
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: @recursive_call_cf
func.func @recursive_call_cf(%arg0: memref<f32>, %cond: i1) {
  // IP:         name = "pre"
  // IP-SAME:    next_access = {{\[}}["pre", "post"]]
  // LOCAL:      name = "pre"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "pre"
  // LC_AR-SAME: next_access = {{\[}}["call", "post"]]
  %0 = memref.load %arg0[] {name = "pre"} : memref<f32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  call @recursive_call_cf(%arg0, %cond) {name = "call"} : (memref<f32>, i1) -> ()
  cf.br ^bb2
^bb2:
  %2 = memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

func.func private @callee1(%arg0: memref<f32>) {
  // IP:         name = "callee1"
  // IP-SAME:    next_access = {{\[}}["post"]]
  // LOCAL:      name = "callee1"
  // LOCAL-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "callee1"} : memref<f32>
  return
}

func.func private @callee2(%arg0: memref<f32>) {
  // IP:         name = "callee2"
  // IP-SAME:    next_access = {{\[}}["post"]]
  // LOCAL:      name = "callee2"
  // LOCAL-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "callee2"} : memref<f32>
  return
}

func.func @conditonal_call(%arg0: memref<f32>, %cond: i1) {
  // IP:         name = "pre"
  // IP-SAME:    next_access = {{\[}}["callee1", "callee2"]]
  // LOCAL:      name = "pre"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "pre"
  // LC_AR-SAME: next_access = {{\[}}["call1", "call2"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  scf.if %cond {
    func.call @callee1(%arg0) {name = "call1"} : (memref<f32>) -> ()
  } else {
    func.call @callee2(%arg0) {name = "call2"} : (memref<f32>) -> ()
  }
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----


// In this test, the "call" operation also accesses %arg0 itself before
// transferring control flow to the callee. Therefore, the order of accesses is
// "caller" -> "call" -> "callee" -> "post"

func.func private @callee(%arg0: memref<f32>) {
  // IP:         name = "callee"
  // IP-SAME:    next_access = {{\[}}["post"]]
  // LOCAL:      name = "callee"
  // LOCAL-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "callee"} : memref<f32>
  return
}

// CHECK-LABEL: @call_and_store_before
func.func @call_and_store_before(%arg0: memref<f32>) {
  // IP:         name = "caller"
  // IP-SAME:    next_access = {{\[}}["call"]]
  // LOCAL:      name = "caller"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "caller"
  // LC_AR-SAME: next_access = {{\[}}["call"]]
  memref.load %arg0[] {name = "caller"} : memref<f32>
  // Note that the access after the entire call is "post".
  // CHECK:      name = "call"
  // CHECK-SAME: next_access = {{\[}}["post"], ["post"]]
  test.call_and_store @callee(%arg0), %arg0 {name = "call", store_before_call = true} : (memref<f32>, memref<f32>) -> ()
  // CHECK:      name = "post"
  // CHECK-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// In this test, the "call" operation also accesses %arg0 itself after getting
// control flow back from the callee. Therefore, the order of accesses is
// "caller" -> "callee" -> "call" -> "post"

func.func private @callee(%arg0: memref<f32>) {
  // IP:         name = "callee"
  // IP-SAME:    next_access = {{\[}}["call"]]
  // LOCAL:      name = "callee"
  // LOCAL-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "callee"} : memref<f32>
  return
}

// CHECK-LABEL: @call_and_store_after
func.func @call_and_store_after(%arg0: memref<f32>) {
  // IP:         name = "caller"
  // IP-SAME:    next_access = {{\[}}["callee"]]
  // LOCAL:      name = "caller"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "caller"
  // LC_AR-SAME: next_access = {{\[}}["call"]]
  memref.load %arg0[] {name = "caller"} : memref<f32>
  // CHECK:      name = "call"
  // CHECK-SAME: next_access = {{\[}}["post"], ["post"]]
  test.call_and_store @callee(%arg0), %arg0 {name = "call", store_before_call = false} : (memref<f32>, memref<f32>) -> ()
  // CHECK:      name = "post"
  // CHECK-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// In this test, the "region" operation also accesses %arg0 itself before
// entering the region. Therefore:
//   - the next access of "pre" is the "region" operation itself;
//   - at the entry of the block, the next access is "post".
// CHECK-LABEL: @store_with_a_region
func.func @store_with_a_region_before(%arg0: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["region"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  // CHECK:              name = "region"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  // CHECK-SAME: next_at_entry_point = {{\[}}{{\[}}["post"]]]
  test.store_with_a_region %arg0 attributes { name = "region", store_before_region = true } {
    test.store_with_a_region_terminator
  } : memref<f32>
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// In this test, the "region" operation also accesses %arg0 itself after
// exiting from the region. Therefore:
//   - the next access of "pre" is the "region" operation itself;
//   - at the entry of the block, the next access is "region".
// CHECK-LABEL: @store_with_a_region
func.func @store_with_a_region_after(%arg0: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["region"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  // CHECK:      name = "region"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  // CHECK-SAME: next_at_entry_point = {{\[}}{{\[}}["region"]]]
  test.store_with_a_region %arg0 attributes { name = "region", store_before_region = false } {
    test.store_with_a_region_terminator
  } : memref<f32>
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// In this test, the operation with a region stores to %arg0 before going to the
// region. Therefore: 
//   - the next access of "pre" is the "region" operation itself;
//   - the next access of the "region" operation (computed as the next access
//     *after* said operation) is the "post" operation;
//   - the next access of the "inner" operation is also "post";
//   - the next access at the entry point of the region of the "region" operation
//     is the "inner" operation.
// That is, the order of access is: "pre" -> "region" -> "inner" -> "post".
// CHECK-LABEL: @store_with_a_region_before_containing_a_load
func.func @store_with_a_region_before_containing_a_load(%arg0: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["region"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  // CHECK:      name = "region"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  // CHECK-SAME: next_at_entry_point = {{\[}}{{\[}}["inner"]]]
  test.store_with_a_region %arg0 attributes { name = "region", store_before_region = true } {
    // CHECK:      name = "inner"
    // CHECK-SAME: next_access = {{\[}}["post"]]
    memref.load %arg0[] {name = "inner"} : memref<f32>
    test.store_with_a_region_terminator
  } : memref<f32>
  // CHECK:      name = "post"
  // CHECK-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// In this test, the operation with a region stores to %arg0 after exiting from
// the region. Therefore:
//   - the next access of "pre" is "inner";
//   - the next access of the "region" operation (computed as the next access
//     *after* said operation) is the "post" operation);
//   - the next access at the entry point of the region of the "region" operation
//     is the "inner" operation;
//   - the next access of the "inner" operation is the "region" operation itself.
// That is, the order of access is "pre" -> "inner" -> "region" -> "post".
// CHECK-LABEL: @store_with_a_region_after_containing_a_load
func.func @store_with_a_region_after_containing_a_load(%arg0: memref<f32>) {
  // CHECK:      name = "pre"
  // CHECK-SAME: next_access = {{\[}}["inner"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  // CHECK:      name = "region"
  // CHECK-SAME: next_access = {{\[}}["post"]]
  // CHECK-SAME: next_at_entry_point = {{\[}}{{\[}}["inner"]]]
  test.store_with_a_region %arg0 attributes { name = "region", store_before_region = false } {
    // CHECK:      name = "inner"
    // CHECK-SAME: next_access = {{\[}}["region"]]
    memref.load %arg0[] {name = "inner"} : memref<f32>
    test.store_with_a_region_terminator
  } : memref<f32>
  // CHECK:      name = "post"
  // CHECK-SAME: next_access = ["unknown"]
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

func.func private @opaque_callee(%arg0: memref<f32>)

// CHECK-LABEL: @call_opaque_callee
func.func @call_opaque_callee(%arg0: memref<f32>) {
  // IP:         name = "pre"
  // IP-SAME:    next_access = ["unknown"]
  // IP_AR:      name = "pre"
  // IP_AR-SAME: next_access = {{\[}}["call"]]
  // LOCAL:      name = "pre"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "pre"
  // LC_AR-SAME: next_access = {{\[}}["call"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  func.call @opaque_callee(%arg0) {name = "call"} : (memref<f32>) -> ()
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: @indirect_call
func.func @indirect_call(%arg0: memref<f32>, %arg1: (memref<f32>) -> ()) {
  // IP:         name = "pre"
  // IP-SAME:    next_access = ["unknown"]
  // IP_AR:      name = "pre"
  // IP_AR-SAME: next_access = ["unknown"] 
  // LOCAL:      name = "pre"
  // LOCAL-SAME: next_access = ["unknown"]
  // LC_AR:      name = "pre"
  // LC_AR-SAME: next_access = {{\[}}["call"]]
  memref.load %arg0[] {name = "pre"} : memref<f32>
  func.call_indirect %arg1(%arg0) {name = "call"} : (memref<f32>) -> ()
  memref.load %arg0[] {name = "post"} : memref<f32>
  return
}
