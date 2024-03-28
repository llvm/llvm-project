// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ptr_test
// CHECK: (%[[ARG0:.*]]: !ptr.ptr, %[[ARG1:.*]]: !ptr.ptr<1 : i32>)
// CHECK: -> (!ptr.ptr<1 : i32>, !ptr.ptr)
func.func @ptr_test(%arg0: !ptr.ptr, %arg1: !ptr.ptr<1 : i32>) -> (!ptr.ptr<1 : i32>, !ptr.ptr) {
  // CHECK: return %[[ARG1]], %[[ARG0]] : !ptr.ptr<1 : i32>, !ptr.ptr
  return %arg1, %arg0 : !ptr.ptr<1 : i32>, !ptr.ptr
}

// -----

// CHECK-LABEL: func @ptr_test
// CHECK: %[[ARG:.*]]: memref<!ptr.ptr>
func.func @ptr_test(%arg0: memref<!ptr.ptr>) {
  return
}

// CHECK-LABEL: func @ptr_test_1
// CHECK: (%[[ARG0:.*]]: !ptr.ptr<#test.const_memory_space>, %[[ARG1:.*]]: !ptr.ptr<#test.const_memory_space<3>>)
func.func @ptr_test_1(%arg0: !ptr.ptr<#test.const_memory_space>,
                      %arg1: !ptr.ptr<#test.const_memory_space<3>>) {
  return
}
