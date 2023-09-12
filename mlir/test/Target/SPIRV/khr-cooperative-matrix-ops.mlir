// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip \
// RUN:  --split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires
  #spirv.vce<v1.5, [Shader, Int8, Int16, Int64, Linkage, CooperativeMatrixKHR],
                   [SPV_KHR_storage_buffer_storage_class, SPV_KHR_cooperative_matrix]> {

  // CHECK-LABEL: @cooperative_matrix_length
  spirv.func @cooperative_matrix_length() "None" {
    // CHECK: {{%.+}} = spirv.KHR.CooperativeMatrixLength : !spirv.coopmatrix<2x2xi32, Subgroup, MatrixB>
    %0 = spirv.KHR.CooperativeMatrixLength : !spirv.coopmatrix<2x2xi32, Subgroup, MatrixB>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_load_1
  spirv.func @cooperative_matrix_load_1(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32) "None" {
    // CHECK:      {{%.+}} = spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <RowMajor>
    // CHECK-SAME:   : !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
    %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor> :
      !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_load_2
  spirv.func @cooperative_matrix_load_2(%ptr : !spirv.ptr<f32, StorageBuffer>, %stride : i64) "None" {
    // CHECK:      {{%.+}} = spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <ColumnMajor>, <Volatile>
    // CHECK-SAME:   : !spirv.ptr<f32, StorageBuffer>, i64 -> !spirv.coopmatrix<8x16xf32, Subgroup, MatrixAcc>
    %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor>, <Volatile> :
      !spirv.ptr<f32, StorageBuffer>, i64 -> !spirv.coopmatrix<8x16xf32, Subgroup, MatrixAcc>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_store_1
  spirv.func @cooperative_matrix_store_1(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32,
                                         %m : !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>) "None" {
    // CHECK:      spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, <RowMajor>
    // CHECK-SAME:   : !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>, i32
    spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, <RowMajor> :
      !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>, i32
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_store_2
  spirv.func @cooperative_matrix_store_2(%ptr : !spirv.ptr<f32, Workgroup>, %stride : i64,
                                         %m : !spirv.coopmatrix<4x8xf32, Subgroup, MatrixB>) "None" {
    // CHECK:      spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, <ColumnMajor>, <Nontemporal>
    // CHECK-SAME:   : !spirv.ptr<f32, Workgroup>, !spirv.coopmatrix<4x8xf32, Subgroup, MatrixB>, i64
    spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, <ColumnMajor>, <Nontemporal> :
      !spirv.ptr<f32, Workgroup>, !spirv.coopmatrix<4x8xf32, Subgroup, MatrixB>, i64
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_muladd
  spirv.func @cooperative_matrix_muladd_1(%a : !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
                                          %b : !spirv.coopmatrix<16x8xi16, Subgroup, MatrixB>,
                                          %c : !spirv.coopmatrix<8x8xi32, Subgroup, MatrixAcc>) "None" {
    // CHECK:      {{%.+}} = spirv.KHR.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}} :
    // CHECK-SAME:   !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
    // CHECK-SAME:   !spirv.coopmatrix<16x8xi16, Subgroup, MatrixB>
    // CHECK-SAME:   -> !spirv.coopmatrix<8x8xi32, Subgroup, MatrixAcc>
    %p = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
                                                        !spirv.coopmatrix<16x8xi16, Subgroup, MatrixB>
                                                        -> !spirv.coopmatrix<8x8xi32, Subgroup, MatrixAcc>

    // CHECK-NEXT: {{%.+}} = spirv.KHR.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}}, <BSigned> :
    // CHECK-SAME:   !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
    // CHECK-SAME:   !spirv.coopmatrix<16x8xi16, Subgroup, MatrixB>
    // CHECK-SAME:   -> !spirv.coopmatrix<8x8xi32, Subgroup, MatrixAcc>
    %q = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c,
                                           <BSigned> : !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
                                                       !spirv.coopmatrix<16x8xi16, Subgroup, MatrixB>
                                                       -> !spirv.coopmatrix<8x8xi32, Subgroup, MatrixAcc>

    // TODO: Handle multiple matrix operands and add relevant testcases here.
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_muladd
  spirv.func @cooperative_matrix_muladd_2(%a : !spirv.coopmatrix<8x8xf32, Workgroup, MatrixA>,
                                          %b : !spirv.coopmatrix<8x8xf32, Workgroup, MatrixB>,
                                          %c : !spirv.coopmatrix<8x8xf32, Workgroup, MatrixAcc>) "None" {
    // CHECK:      {{%.+}} = spirv.KHR.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}} :
    // CHECK-SAME:   !spirv.coopmatrix<8x8xf32, Workgroup, MatrixA>,
    // CHECK-SAME:   !spirv.coopmatrix<8x8xf32, Workgroup, MatrixB>
    // CHECK-SAME:   -> !spirv.coopmatrix<8x8xf32, Workgroup, MatrixAcc>
    %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<8x8xf32, Workgroup, MatrixA>,
                                                        !spirv.coopmatrix<8x8xf32, Workgroup, MatrixB>
                                                        -> !spirv.coopmatrix<8x8xf32, Workgroup, MatrixAcc>

    spirv.Return
  }

}
