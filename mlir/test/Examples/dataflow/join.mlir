// RUN: dataflow-opt %s -test-metadata-analysis -split-input-file | FileCheck %s

//      CHECK: {{likes_pizza: true}}
// CHECK-NEXT: {{likes_pizza: true}} 
func.func @single_join(%arg0 : index, %arg1 : index) -> index {
  %1 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = true }} : index
  %2 = arith.addi %1, %arg1 : index
  return %2 : index
}

// -----

//      CHECK: {{likes_pizza: true}}
// CHECK-NEXT: {{likes_hotdog: true}}
// CHECK-NEXT: {{likes_pizza: true, likes_hotdog: true}}
func.func @muti_join(%arg0 : index, %arg1 : index) -> index {
  %1 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = true }} : index
  %2 = arith.addi %arg0, %arg1 {metadata = { likes_hotdog = true }} : index
  %3 = arith.addi %1, %2 : index
  return %3 : index
}

// -----

//      CHECK: {{likes_pizza: true}}
// CHECK-NEXT: {{likes_pizza: false}}
// CHECK-NEXT: {{likes_pizza: true}}
func.func @conflict_join(%arg0 : index, %arg1 : index) -> index {
  %1 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = true }} : index 
  %2 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = false }} : index
  %3 = arith.addi %1, %2 : index
  return %3 : index
}
