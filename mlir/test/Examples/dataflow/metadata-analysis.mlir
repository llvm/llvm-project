// RUN: dataflow-opt %s -test-metadata-analysis -split-input-file | FileCheck %s

//      CHECK: {{.*}} = arith.addi {{.*}}, {{.*}} {metadata = {likes_pizza = true}} : index
//      CHECK: {{"likes_pizza": true}}
// CHECK-NEXT: {{.*}} = arith.addi {{.*}}, {{.*}} : index
//      CHECK: {{"likes_pizza": true}}
func.func @single_join(%arg0 : index, %arg1 : index) -> index {
  %1 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = true }} : index
  %2 = arith.addi %1, %arg1 : index
  return %2 : index
}

// -----

//      CHECK: {{.*}} = arith.addi {{.*}}, {{.*}} {metadata = {likes_pizza = true}} : index
//      CHECK: {{"likes_pizza": true}}
// CHECK-NEXT: {{.*}} = arith.addi {{.*}}, {{.*}} {metadata = {likes_hotdog = true}} : index
//      CHECK: {{"likes_hotdog": true}}
// CHECK-NEXT: {{.*}} = arith.addi {{.*}}, {{.*}} : index
//      CHECK: {{"likes_hotdog": true, "likes_pizza": true}}
func.func @muti_join(%arg0 : index, %arg1 : index) -> index {
  %1 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = true }} : index
  %2 = arith.addi %arg0, %arg1 {metadata = { likes_hotdog = true }} : index
  %3 = arith.addi %1, %2 : index
  return %3 : index
}

// -----

//      CHECK: {{.*}} = arith.addi {{.*}}, {{.*}} {metadata = {likes_pizza = true}} : index
//      CHECK: {{"likes_pizza": true}}
// CHECK-NEXT: {{.*}} = arith.addi {{.*}}, {{.*}} {metadata = {likes_pizza = false}} : index
//      CHECK: {{"likes_pizza": false}}
// CHECK-NEXT: {{.*}} = arith.addi {{.*}}, {{.*}} : index
//      CHECK: {{"likes_pizza": true}}

func.func @conflict_join(%arg0 : index, %arg1 : index) -> index {
  %1 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = true }} : index 
  %2 = arith.addi %arg0, %arg1 {metadata = { likes_pizza = false }} : index
  %3 = arith.addi %1, %2 : index
  return %3 : index
}
