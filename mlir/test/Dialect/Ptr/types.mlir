// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ptr_test
// CHECK: (%[[ARG0:.*]]: !ptr.ptr<#test.const_memory_space>, %[[ARG1:.*]]: !ptr.ptr<#test.const_memory_space<1>>)
// CHECK: -> (!ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>)
func.func @ptr_test(%arg0: !ptr.ptr<#test.const_memory_space>, %arg1: !ptr.ptr<#test.const_memory_space<1>>) -> (!ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>) {
  // CHECK: return %[[ARG1]], %[[ARG0]] : !ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>
  return %arg1, %arg0 : !ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>
}

// -----

// CHECK-LABEL: func @ptr_test
// CHECK: %[[ARG:.*]]: memref<!ptr.ptr<#test.const_memory_space>>
func.func @ptr_test(%arg0: memref<!ptr.ptr<#test.const_memory_space>>) {
  return
}

// CHECK-LABEL: func @ptr_test_1
// CHECK: (%[[ARG0:.*]]: !ptr.ptr<#test.const_memory_space>, %[[ARG1:.*]]: !ptr.ptr<#test.const_memory_space<3>>)
func.func @ptr_test_1(%arg0: !ptr.ptr<#test.const_memory_space>,
                      %arg1: !ptr.ptr<#test.const_memory_space<3>>) {
  return
}
