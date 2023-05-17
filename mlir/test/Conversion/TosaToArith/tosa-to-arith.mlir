// RUN: mlir-opt --split-input-file --tosa-to-arith="include-apply-rescale=true use-32-bit=true" %s -verify-diagnostics -o -| FileCheck %s
// RUN: mlir-opt --split-input-file --tosa-to-arith="include-apply-rescale=false" %s -verify-diagnostics -o -| FileCheck --check-prefix="SCALE" %s

// CHECK-LABEL: func @const_test
func.func @const_test() -> (tensor<i32>) {
  // CHECK: [[C3:%.+]] = arith.constant dense<3> : tensor<i32>
  %result = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>

  // CHECK: return [[C3]]
  return %result : tensor<i32>
}

// -----

// CHECK-LABEL: @apply_scale_test_i32
// SCALE: "tosa.apply_scale"
func.func @apply_scale_test_i32(%arg0 : i32, %arg1 : i32, %arg2 : i8) -> (i32) {
  // CHECK-DAG: %[[S32:.+]] = arith.extui %arg2 : i8 to i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-DAG: %[[C30:.+]] = arith.constant 30 : i32
  // CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32

  // Compute the high-low values of the matmul in 64-bits.
  // CHECK-DAG: %[[LOW:.+]], %[[HI:.+]] = arith.mulsi_extended %arg0, %arg1

  // Determine whether the high bits need to shift left or right and by how much.
  // CHECK-DAG: %[[OVER31:.+]] = arith.cmpi sge, %[[S32]], %[[C32]]
  // CHECK-DAG: %[[OVER32:.+]] = arith.cmpi sgt, %[[S32]], %[[C32]]
  // CHECK-DAG: %[[HISHLN:.+]] = arith.subi %[[C32]], %[[S32]]
  // CHECK-DAG: %[[HISHRN:.+]] = arith.subi %[[S32]], %[[C32]]
  // CHECK-DAG: %[[HISHL:.+]] = arith.select %[[OVER31]], %[[C0]], %[[HISHLN]]
  // CHECK-DAG: %[[HISHR:.+]] = arith.select %[[OVER31]], %[[HISHRN]], %[[C0]]

  // Apply double rounding.
  // CHECK-DAG: %[[CN1:.+]] = arith.constant -1
  // CHECK-DAG: %[[POS:.+]] = arith.cmpi sge, %arg0, %[[C0]]
  // CHECK-DAG: %[[DIR:.+]] = arith.select %[[POS]], %[[C1]], %[[CN1]]
  // CHECK-DAG: %[[DRND:.+]] = arith.select %[[OVER31]], %[[DIR]], %[[C0]]
  // CHECK-DAG: %[[DSHFTR:.+]] = arith.shrui %[[LOW]], %[[C30]]
  // CHECK-DAG: %[[DRNDED:.+]] = arith.addi %[[DSHFTR]], %[[DRND]]
  // CHECK-DAG: %[[DCARRY:.+]] = arith.shrsi %[[DRNDED]], %[[C2:.+]]
  // CHECK-DAG: %[[DBIT:.+]] = arith.shli %[[DRND]], %[[C30]]
  // CHECK-DAG: %[[DLOW:.+]] = arith.addi %[[LOW]], %[[DBIT]]
  // CHECK-DAG: %[[DHI:.+]] = arith.addi %[[HI]], %[[DCARRY]]

  // Apply low-bit rounding.
  // CHECK-DAG: %[[SHFTM1:.+]] = arith.subi %[[S32]], %[[C1]]
  // CHECK-DAG: %[[LBIT:.+]] = arith.shli %[[C1]], %[[SHFTM1]]
  // CHECK-DAG: %[[HALF:.+]] = arith.select %[[OVER32]], %[[C0]], %[[LBIT]]
  // CHECK-DAG: %[[LADD:.+]] = arith.addi %[[DLOW]], %[[HALF]]
  // CHECK-DAG: %[[LLO:.+]] = arith.cmpi ugt, %[[DLOW]], %[[LADD]]
  // CHECK-DAG: %[[LCARRY:.+]] = arith.extui %[[LLO]] : i1 to i32
  // CHECK-DAG: %[[LRNDED:.+]] = arith.addi %[[DHI]], %[[LCARRY]]

  // Apply high-bit rounding.
  // CHECK-DAG: %[[HISHRM1:.+]] = arith.subi %[[HISHR]], %[[C1]]
  // CHECK-DAG: %[[LHISHFT:.+]] = arith.shli %[[C1]], %[[HISHRM1]]
  // CHECK-DAG: %[[LHI:.+]] = arith.select %[[OVER32]], %[[LHISHFT]], %[[C0]]
  // CHECK-DAG: %[[FHI:.+]] = arith.addi %[[LRNDED]], %[[LHI]]

  // Combine hi-low into the final result.
  // CHECK-DAG: %[[HIL:.+]] = arith.shli %[[FHI]], %[[HISHL]]
  // CHECK-DAG: %[[HIALIGN:.+]] = arith.shrsi %[[HIL:.+]], %[[HISHR]]
  // CHECK-DAG: %[[LOR:.+]] = arith.shrui %[[LADD]], %[[S32]]
  // CHECK-DAG: %[[LOWALIGN:.+]] = arith.select %[[OVER31]], %[[C0]], %[[LOR]]
  // CHECK-DAG: %[[RESULT:.+]] = arith.addi %[[LOWALIGN]], %[[HIALIGN]]
  // CHECK: return %[[RESULT]]
  %res = "tosa.apply_scale"(%arg0, %arg1, %arg2) {double_round = true} : (i32, i32, i8) -> i32
  return %res : i32
}

// -----

// CHECK-LABEL: @apply_scale_test_vector
// SCALE: "tosa.apply_scale"
func.func @apply_scale_test_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>, %arg2 : vector<4xi8>) -> (vector<4xi32>) {
  // CHECK-NOT: "tosa.apply_scale"
  %res = "tosa.apply_scale"(%arg0, %arg1, %arg2) {double_round = true} : (vector<4xi32>, vector<4xi32>, vector<4xi8>) -> vector<4xi32>
  return %res : vector<4xi32>
}

// -----

// CHECK-LABEL: @apply_scale_test_i48
// SCALE: "tosa.apply_scale"
func.func @apply_scale_test_i48(%arg0 : i48, %arg1 : i32, %arg2 : i8) -> (i32) {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i48
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
  // CHECK-DAG: %[[C31:.+]] = arith.constant 31 : i32

  // Multiply in 64 bits.
  // CHECK-DAG: %[[V64:.+]] = arith.extsi %arg0 : i48 to i64
  // CHECK-DAG: %[[M64:.+]] = arith.extsi %arg1 : i32 to i64
  // CHECK-DAG: %[[MUL:.+]] = arith.muli %[[V64]], %[[M64]]

  // Round normally.
  // CHECK-DAG: %[[S32:.+]] = arith.extui %arg2 : i8 to i32
  // CHECK-DAG: %[[S64:.+]] = arith.extui %[[S32]] : i32 to i64
  // CHECK-DAG: %[[ONEL:.+]] = arith.shli %[[C1]], %[[S64]] : i64
  // CHECK-DAG: %[[ONER:.+]] = arith.shrui %[[ONEL]], %[[C1]]
  // CHECK-DAG: %[[ROUND:.+]] = arith.addi %[[MUL]], %[[ONER]]

  // Apply double rounding.
  // CHECK-DAG: %[[DUP:.+]] = arith.constant 1073741824 : i64
  // CHECK-DAG: %[[DDOWN:.+]] = arith.constant -1073741824 : i64
  // CHECK-DAG: %[[POS:.+]] = arith.cmpi sge, %arg0, %[[C0]]
  // CHECK-DAG: %[[DBIT:.+]] = arith.select %[[POS]], %[[DUP]], %[[DDOWN]]
  // CHECK-DAG: %[[DRND:.+]] = arith.addi %[[DBIT]], %[[ROUND]]
  // CHECK-DAG: %[[USED:.+]] = arith.cmpi sgt, %[[S32]], %[[C31]] : i32
  // CHECK-DAG: %[[RES64:.+]] = arith.select %[[USED]], %[[DRND]], %[[ROUND]] : i64

  // Shift and truncate final answer.
  // CHECK-DAG: %[[SHR:.+]] = arith.shrsi %[[RES64]], %[[S64]]
  // CHECK-DAG: %[[TRUNC:.+]] = arith.trunci %[[SHR]] : i64 to i32
  // CHECK: return %[[TRUNC]]
  %res = "tosa.apply_scale"(%arg0, %arg1, %arg2) {double_round = true} : (i48, i32, i8) -> i32
  return %res : i32
}
