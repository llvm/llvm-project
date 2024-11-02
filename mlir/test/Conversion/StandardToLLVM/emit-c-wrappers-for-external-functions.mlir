// RUN: mlir-opt -convert-func-to-llvm='emit-c-wrappers=1' %s | FileCheck %s

// CHECK: llvm.func @res_attrs_with_memref_return() -> (!llvm.struct{{.*}} {test.returnOne})
// CHECK-LABEL: llvm.func @_mlir_ciface_res_attrs_with_memref_return
// CHECK: !llvm.ptr{{.*}} {test.returnOne}
func.func private @res_attrs_with_memref_return() -> (memref<f32> {test.returnOne})

// CHECK: llvm.func @res_attrs_with_value_return() -> (f32 {test.returnOne = 1 : i64})
// CHECK-LABEL: llvm.func @_mlir_ciface_res_attrs_with_value_return
// CHECK: -> (f32 {test.returnOne = 1 : i64})
func.func private @res_attrs_with_value_return() -> (f32 {test.returnOne = 1})

// CHECK: llvm.func @multiple_return() -> (!llvm.struct<{{.*}}> {llvm.struct_attrs = [{test.returnOne = 1 : i64}, {test.returnThree = 3 : i64, test.returnTwo = 2 : i64}]})
// CHECK-LABEL: llvm.func @_mlir_ciface_multiple_return
// CHECK: (!llvm.ptr<{{.*}}> {llvm.struct_attrs = [{test.returnOne = 1 : i64}, {test.returnThree = 3 : i64, test.returnTwo = 2 : i64}]})
func.func private @multiple_return() -> (memref<f32> {test.returnOne = 1}, f32 {test.returnTwo = 2, test.returnThree = 3})

// CHECK: llvm.func @multiple_return_missing_res_attr() -> (!llvm.struct<{{.*}}> {llvm.struct_attrs = [{test.returnOne = 1 : i64}, {}, {test.returnThree = 3 : i64, test.returnTwo = 2 : i64}]})
// CHECK-LABEL: llvm.func @_mlir_ciface_multiple_return_missing_res_attr
// CHECK: (!llvm.ptr<{{.*}}> {llvm.struct_attrs = [{test.returnOne = 1 : i64}, {}, {test.returnThree = 3 : i64, test.returnTwo = 2 : i64}]})
func.func private @multiple_return_missing_res_attr() -> (memref<f32> {test.returnOne = 1}, i64, f32 {test.returnTwo = 2, test.returnThree = 3})

// CHECK: llvm.func @one_arg_attr_no_res_attrs_with_memref_return({{.*}}) -> !llvm.struct{{.*}}
// CHECK-LABEL: llvm.func @_mlir_ciface_one_arg_attr_no_res_attrs_with_memref_return
// CHECK: !llvm.ptr<{{.*}}>, !llvm.ptr<{{.*}}> {test.argOne = 1 : i64}
func.func private @one_arg_attr_no_res_attrs_with_memref_return(%arg0: memref<f32> {test.argOne = 1}) -> memref<f32>

// CHECK: llvm.func @one_arg_attr_one_res_attr_with_memref_return({{.*}}) -> (!llvm.struct<{{.*}}> {test.returnOne = 1 : i64})
// CHECK-LABEL: llvm.func @_mlir_ciface_one_arg_attr_one_res_attr_with_memref_return
// CHECK: (!llvm.ptr<{{.*}}> {test.returnOne = 1 : i64}, !llvm.ptr<{{.*}}> {test.argOne = 1 : i64}
func.func private @one_arg_attr_one_res_attr_with_memref_return(%arg0: memref<f32> {test.argOne = 1}) -> (memref<f32> {test.returnOne = 1})

// CHECK: llvm.func @one_arg_attr_one_res_attr_with_value_return({{.*}}) -> (f32 {test.returnOne = 1 : i64})
// CHECK-LABEL: llvm.func @_mlir_ciface_one_arg_attr_one_res_attr_with_value_return
// CHECK: (!llvm.ptr<{{.*}}> {test.argOne = 1 : i64}) -> (f32 {test.returnOne = 1 : i64})
func.func private @one_arg_attr_one_res_attr_with_value_return(%arg0: memref<f32> {test.argOne = 1}) -> (f32 {test.returnOne = 1})

// CHECK: llvm.func @multiple_arg_attr_multiple_res_attr({{.*}}) -> (!llvm.struct<{{.*}}> {llvm.struct_attrs = [{}, {test.returnOne = 1 : i64}, {test.returnTwo = 2 : i64}]})
// CHECK-LABEL: llvm.func @_mlir_ciface_multiple_arg_attr_multiple_res_attr
// CHECK: (!llvm.ptr<{{.*}}> {llvm.struct_attrs = [{}, {test.returnOne = 1 : i64}, {test.returnTwo = 2 : i64}]}, !llvm.ptr<{{.*}}> {test.argZero = 0 : i64}, f32, i32 {test.argTwo = 2 : i64}
func.func private @multiple_arg_attr_multiple_res_attr(%arg0: memref<f32> {test.argZero = 0}, %arg1: f32, %arg2: i32 {test.argTwo = 2}) -> (f32, memref<i32> {test.returnOne = 1}, i32 {test.returnTwo = 2})
