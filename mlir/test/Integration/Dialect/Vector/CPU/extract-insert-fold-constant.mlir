// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %v = arith.constant dense<0> : vector<2x2xi32>
  %c_0 = arith.constant 0 : index
  %c_1 = arith.constant 1 : index
  %i32_0 = arith.constant 0 : i32
  %i32_1 = arith.constant 1 : i32
  %i32_2 = arith.constant 2 : i32
  %i32_3 = arith.constant 3 : i32
  %v_1 = vector.insert %i32_0, %v[%c_0, %c_0] : i32 into vector<2x2xi32>
  %v_2 = vector.insert %i32_1, %v_1[%c_0, %c_1] : i32 into vector<2x2xi32>
  %v_3 = vector.insert %i32_2, %v_2[%c_1, %c_0] : i32 into vector<2x2xi32>
  %v_4 = vector.insert %i32_3, %v_3[%c_1, %c_1] : i32 into vector<2x2xi32>
  // CHECK: ( ( 0, 1 ), ( 2, 3 ) ) 
  vector.print %v_4 : vector<2x2xi32>
  %v_5 = arith.constant dense<[[0, 1], [2, 3]]> : vector<2x2xi32>
  // CHECK: 0
  %i32_4 = vector.extract %v_5[%c_0, %c_0] : i32 from vector<2x2xi32>
  // CHECK: 1
  %i32_5 = vector.extract %v_5[%c_0, %c_1] : i32 from vector<2x2xi32>
  // CHECK: 2
  %i32_6 = vector.extract %v_5[%c_1, %c_0] : i32 from vector<2x2xi32>
  // CHECK: 3
  %i32_7 = vector.extract %v_5[%c_1, %c_1] : i32 from vector<2x2xi32>
  vector.print %i32_4 : i32
  vector.print %i32_5 : i32
  vector.print %i32_6 : i32
  vector.print %i32_7 : i32
  return
}
