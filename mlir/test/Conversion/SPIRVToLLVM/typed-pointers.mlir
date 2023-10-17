// RUN: mlir-opt -split-input-file -convert-spirv-to-llvm='use-opaque-pointers=0' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @pointer_scalar(!llvm.ptr<i1>, !llvm.ptr<f32>)
spirv.func @pointer_scalar(!spirv.ptr<i1, Uniform>, !spirv.ptr<f32, Private>) "None"

// CHECK-LABEL: @pointer_vector(!llvm.ptr<vector<4xi32>>)
spirv.func @pointer_vector(!spirv.ptr<vector<4xi32>, Function>) "None"

// CHECK-LABEL: @bitcast_pointer
spirv.func @bitcast_pointer(%arg0: !spirv.ptr<f32, Function>) "None" {
  // CHECK: llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<i32>
  %0 = spirv.Bitcast %arg0 : !spirv.ptr<f32, Function> to !spirv.ptr<i32, Function>
  spirv.Return
}
