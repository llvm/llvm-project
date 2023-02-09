// RUN: mlir-opt -llvm-request-c-wrappers -convert-func-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

// CHECK: llvm.func @res_attrs_with_memref_return() -> (!llvm.struct{{.*}} {test.returnOne})
// CHECK-LABEL: llvm.func @_mlir_ciface_res_attrs_with_memref_return
// CHECK-NOT: test.returnOne
func.func @res_attrs_with_memref_return() -> (memref<f32> {test.returnOne}) {
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// CHECK: llvm.func @res_attrs_with_value_return() -> (f32 {test.returnOne = 1 : i64})
// CHECK-LABEL: llvm.func @_mlir_ciface_res_attrs_with_value_return
// CHECK-SAME: -> (f32 {test.returnOne = 1 : i64})
func.func @res_attrs_with_value_return() -> (f32 {test.returnOne = 1}) {
  %0 = arith.constant 1.00 : f32
  return %0 : f32
}

// CHECK: llvm.func @multiple_return() -> !llvm.struct<{{.*}}>
// CHECK-LABEL: llvm.func @_mlir_ciface_multiple_return
// CHECK-SAME: !llvm.ptr
// CHECK-NOT: test.returnOne
// CHECK-NOT: test.returnTwo
// CHECK-NOT: test.returnThree
func.func @multiple_return() -> (memref<f32> {test.returnOne = 1}, f32 {test.returnTwo = 2, test.returnThree = 3}) {
  %0 = memref.alloc() : memref<f32>
  %1 = arith.constant 1.00 : f32
  return %0, %1 : memref<f32>, f32
}

// CHECK: llvm.func @multiple_return_missing_res_attr() -> !llvm.struct<{{.*}}>
// CHECK-LABEL: llvm.func @_mlir_ciface_multiple_return_missing_res_attr
// CHECK-SAME: !llvm.ptr
// CHECK-NOT: test.returnOne
// CHECK-NOT: test.returnTwo
// CHECK-NOT: test.returnThree
func.func @multiple_return_missing_res_attr() -> (memref<f32> {test.returnOne = 1}, i64, f32 {test.returnTwo = 2, test.returnThree = 3}) {
  %0 = memref.alloc() : memref<f32>
  %1 = arith.constant 2 : i64
  %2 = arith.constant 1.00 : f32
  return %0, %1, %2 : memref<f32>, i64, f32
}

// CHECK: llvm.func @one_arg_attr_no_res_attrs_with_memref_return({{.*}}) -> !llvm.struct{{.*}}
// CHECK-LABEL: llvm.func @_mlir_ciface_one_arg_attr_no_res_attrs_with_memref_return
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: {test.argOne = 1 : i64})
func.func @one_arg_attr_no_res_attrs_with_memref_return(%arg0: memref<f32> {test.argOne = 1}) -> memref<f32> {
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// CHECK: llvm.func @one_arg_attr_one_res_attr_with_memref_return({{.*}}) -> (!llvm.struct<{{.*}}> {test.returnOne = 1 : i64})
// CHECK-LABEL: llvm.func @_mlir_ciface_one_arg_attr_one_res_attr_with_memref_return
// CHECK-SAME: !llvm.ptr
// CHECK-NOT: test.returnOne
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: {test.argOne = 1 : i64})
func.func @one_arg_attr_one_res_attr_with_memref_return(%arg0: memref<f32> {test.argOne = 1}) -> (memref<f32> {test.returnOne = 1}) {
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// CHECK: llvm.func @one_arg_attr_one_res_attr_with_value_return({{.*}}) -> (f32 {test.returnOne = 1 : i64})
// CHECK-LABEL: llvm.func @_mlir_ciface_one_arg_attr_one_res_attr_with_value_return
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: {test.argOne = 1 : i64})
// CHECK-SAME: -> (f32 {test.returnOne = 1 : i64})
func.func @one_arg_attr_one_res_attr_with_value_return(%arg0: memref<f32> {test.argOne = 1}) -> (f32 {test.returnOne = 1}) {
  %0 = arith.constant 1.00 : f32
  return %0 : f32
}

// CHECK: llvm.func @multiple_arg_attr_multiple_res_attr({{.*}}) -> !llvm.struct<{{.*}}>
// CHECK-LABEL: llvm.func @_mlir_ciface_multiple_arg_attr_multiple_res_attr
// CHECK-SAME: !llvm.ptr
// CHECK-NOT: test.returnOne
// CHECK-NOT: test.returnTwo
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: {test.argZero = 0 : i64}
// CHECK-SAME: f32
// CHECK-SAME: i32 {test.argTwo = 2 : i64})
func.func @multiple_arg_attr_multiple_res_attr(%arg0: memref<f32> {test.argZero = 0}, %arg1: f32, %arg2: i32 {test.argTwo = 2}) -> (f32, memref<i32> {test.returnOne = 1}, i32 {test.returnTwo = 2}) {
  %0 = arith.constant 1.00 : f32
  %1 = memref.alloc() : memref<i32>
  %2 = arith.constant 2 : i32
  return %0, %1, %2 : f32, memref<i32>, i32
}

// CHECK: llvm.func @drop_linkage_attr() -> (!llvm.struct{{.*}} {test.returnOne})
// CHECK-LABEL: llvm.func @_mlir_ciface_drop_linkage_attr
// CHECK-SAME: !llvm.ptr
// CHECK-NOT: llvm.linkage
func.func @drop_linkage_attr() -> (memref<f32> {test.returnOne}) attributes { llvm.linkage = #llvm.linkage<external> } {
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}
