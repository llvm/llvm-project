// RUN: mlir-opt -test-generic-ir-visitors -allow-unregistered-dialect -split-input-file %s | FileCheck %s
// RUN: mlir-opt -test-generic-ir-visitors-interrupt -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Verify the different configurations of generic IR visitors.

func @structured_cfg() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c1 to %c10 step %c1 {
    %cond = "use0"(%i) : (index) -> (i1)
    scf.if %cond {
      "use1"(%i) : (index) -> ()
    } else {
      "use2"(%i) : (index) -> ()
    }
    "use3"(%i) : (index) -> ()
  }
  return
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 1 op 'builtin.func' before all regions
// CHECK: step 2 op 'arith.constant' before all regions
// CHECK: step 3 op 'arith.constant' before all regions
// CHECK: step 4 op 'arith.constant' before all regions
// CHECK: step 5 op 'scf.for' before all regions
// CHECK: step 6 op 'use0' before all regions
// CHECK: step 7 op 'scf.if' before all regions
// CHECK: step 8 op 'use1' before all regions
// CHECK: step 9 op 'scf.yield' before all regions
// CHECK: step 10 op 'scf.if' before region #1
// CHECK: step 11 op 'use2' before all regions
// CHECK: step 12 op 'scf.yield' before all regions
// CHECK: step 13 op 'scf.if' after all regions
// CHECK: step 14 op 'use3' before all regions
// CHECK: step 15 op 'scf.yield' before all regions
// CHECK: step 16 op 'scf.for' after all regions
// CHECK: step 17 op 'std.return' before all regions
// CHECK: step 18 op 'builtin.func' after all regions
// CHECK: step 19 op 'builtin.module' after all regions

// -----
// Test the specific operation type visitor.

func @correct_number_of_regions() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c1 to %c10 step %c1 {
    "test.two_region_op"()(
      {"work"() : () -> ()},
      {"work"() : () -> ()}
    ) : () -> ()
  }
  return
}

// CHECK: step 0 op 'builtin.module' before all regions
// CHECK: step 15 op 'builtin.module' after all regions
// CHECK: step 16 op 'test.two_region_op' before all regions
// CHECK: step 17 op 'test.two_region_op' before region #1
// CHECK: step 18 op 'test.two_region_op' after all regions
