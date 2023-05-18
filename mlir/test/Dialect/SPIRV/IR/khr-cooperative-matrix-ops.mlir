// RUN: mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// CooperativeMatrix (KHR) extension ops.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cooperative_matrix_length
spirv.func @cooperative_matrix_length() -> i32 "None" {
  // CHECK: {{%.*}} = spirv.KHR.CooperativeMatrixLength : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
  %0 = spirv.KHR.CooperativeMatrixLength : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
  spirv.ReturnValue %0 : i32
}

// -----

// CHECK-LABEL: @cooperative_matrix_load
spirv.func @cooperative_matrix_load(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32) "None" {
  // CHECK:      {{%.*}} = spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <RowMajor> :
  // CHECK-SAME:   !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor> :
    !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_load_memoperand
spirv.func @cooperative_matrix_load_memoperand(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32) "None" {
  // CHECK:      {{%.*}} = spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <ColumnMajor>, <Volatile> :
  // CHECK-SAME:   !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor>, <Volatile> :
    !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_load_vector_ptr_type
spirv.func @cooperative_matrix_load_vector_ptr_type(%ptr : !spirv.ptr<vector<4xi32>, StorageBuffer>, %stride : i32) "None" {
  // CHECK:      {{%.*}} = spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <RowMajor>, <Volatile> :
  // CHECK-SAME:   !spirv.ptr<vector<4xi32>, StorageBuffer>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor>, <Volatile> :
    !spirv.ptr<vector<4xi32>, StorageBuffer>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_load_function
spirv.func @cooperative_matrix_load_function(%ptr : !spirv.ptr<i32, Function>, %stride : i32) "None" {
  // CHECK:      {{%.*}} = spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <RowMajor> :
  // CHECK-SAME:   !spirv.ptr<i32, Function>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixAcc>
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor> :
    !spirv.ptr<i32, Function>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_load_stride_i16
spirv.func @cooperative_matrix_load_stride_i16(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i16) "None" {
  // CHECK:      {{%.*}} = spirv.KHR.CooperativeMatrixLoad {{%.*}}, {{%.*}}, <RowMajor> :
  // CHECK-SAME:   !spirv.ptr<i32, StorageBuffer>, i16 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor> :
    !spirv.ptr<i32, StorageBuffer>, i16 -> !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_store
spirv.func @cooperative_matrix_store(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32,
                                     %m : !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>) "None" {
  // CHECK:      spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}}, <RowMajor> :
  // CHECK-SAME:   !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>, i32
  spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, <RowMajor> :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>, i32
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_store_memoperand
spirv.func @cooperative_matrix_store_memoperand(%ptr : !spirv.ptr<i32, StorageBuffer>,
                                                %m : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
                                                %stride : i32) "None" {
  // CHECK:       spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}}, <ColumnMajor>, <Volatile> :
  // CHECK-SAME:    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>, i32
  spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, <ColumnMajor>, <Volatile> :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>, i32
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_store_stride_i16
spirv.func @cooperative_matrix_store_stride_i16(%ptr : !spirv.ptr<i32, StorageBuffer>,
                                                %m : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
                                                %stride : i16) "None" {
  // CHECK:       spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}}, <ColumnMajor> :
  // CHECK-SAME:    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>, i16
  spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, <ColumnMajor> :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>, i16
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_load_bad_ptr(%ptr : !spirv.ptr<!spirv.struct<(f32 [0])>, StorageBuffer>, %stride : i32) "None" {
  // expected-error @+1 {{Pointer must point to a scalar or vector type}}
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor> :
    !spirv.ptr<!spirv.struct<(f32 [0])>, StorageBuffer>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_load_missing_attr(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32) "None" {
  // expected-error @+1 {{expected ','}}
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride :
    !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_load_bad_operad(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32) "None" {
  // expected-error @+1 {{op not compatible with memory operand 'MakePointerAvailable'}}
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor>, <MakePointerAvailable> :
    !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_load_aligned(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32) "None" {
  // expected-error @+1 {{op has unhandled memory operand 'Aligned'}}
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor>, <Aligned> :
    !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_load_aligned(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32) "None" {
  // expected-error @+1 {{op has unhandled memory operand 'Aligned'}}
  %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor>, <Volatile|Aligned> :
    !spirv.ptr<i32, StorageBuffer>, i32 -> !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_store_missing_attr(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32,
                                                  %m : !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>) "None" {
  // expected-error @+1 {{expected ','}}
  spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_store_missing_attr(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32,
                                                  %m : !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>) "None" {
  // expected-error @+1 {{expected '<'}}
  spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>, i32
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_store_bad_object_type(%ptr : !spirv.ptr<i32, StorageBuffer>,
                                                     %stride : i32) "None" {
  // expected-error @+1 {{op operand #1 must be any SPIR-V cooperative matrix type}}
  spirv.KHR.CooperativeMatrixStore %ptr, %stride, %stride, <RowMajor> :
    !spirv.ptr<i32, StorageBuffer>, i32, i32
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_store_bad_operand(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32,
                                                 %m : !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>) "None" {
  // expected-error @+1 {{op not compatible with memory operand 'MakePointerVisible'}}
  spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, <RowMajor>, <MakePointerVisible> :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>, i32
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_store(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32,
                                     %m : !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>) "None" {
  // expected-error @+1 {{op has unhandled memory operand 'Aligned'}}
  spirv.KHR.CooperativeMatrixStore %ptr, %m, %stride, <RowMajor>, <Aligned> :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup, MatrixA>, i32
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

spirv.func @cooperative_matrix_muladd_matrix_operands(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>,
                                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  %p = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c, <AccSat> :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  %q = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c, <ASigned | BSigned> :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c, <ASigned | BSigned | AccSat> :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

spirv.func @cooperative_matrix_muladd_f32(%a : !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>,
                                          %b : !spirv.coopmatrix<4x4xf32, Subgroup, MatrixB>,
                                          %c : !spirv.coopmatrix<4x4xf32, Subgroup, MatrixAcc>) "None" {
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>,
        !spirv.coopmatrix<4x4xf32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<4x4xf32, Subgroup, MatrixAcc>
  spirv.Return
}

spirv.func @cooperative_matrix_muladd_i8_i32(%a : !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
                                             %b : !spirv.coopmatrix<16x4xi8, Subgroup, MatrixB>,
                                             %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi8, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

spirv.func @cooperative_matrix_muladd_i8_i16_i32(%a : !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
                                                 %b : !spirv.coopmatrix<16x4xi16, Subgroup, MatrixB>,
                                                 %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi8, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi16, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

spirv.func @cooperative_matrix_muladd_workgroup(%a : !spirv.coopmatrix<4x4xf16, Workgroup, MatrixA>,
                                                %b : !spirv.coopmatrix<4x4xf16, Workgroup, MatrixB>,
                                                %c : !spirv.coopmatrix<4x4xf16, Workgroup, MatrixAcc>) "None" {
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<4x4xf16, Workgroup, MatrixA>,
        !spirv.coopmatrix<4x4xf16, Workgroup, MatrixB> ->
          !spirv.coopmatrix<4x4xf16, Workgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op operand #0 must be of use 'MatrixA'}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>) "None" {
  // expected-error @+1 {{expected ','}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>) "None" {
  // expected-error @+1 {{expected SSA operand}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, <ASigned> :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixB>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{expected '<'}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixA>,
                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op operand #1 must be of use 'MatrixB'}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixA> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixB>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op operand #2 must be of use 'MatrixAcc'}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixB>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<10x4xi32, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op matrix size mismatch on dimension 'M'}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<10x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<4x16xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op matrix size mismatch on dimension 'N'}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<4x16xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<12x4xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op matrix size mismatch on dimension 'K'}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<12x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                      %b : !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB>,
                                      %c : !spirv.coopmatrix<8x4xi32, Workgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op matrix scope mismatch}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c :
        !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xi32, Workgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd_matrix_operands(%a : !spirv.coopmatrix<8x16xf16, Subgroup, MatrixA>,
                                                      %b : !spirv.coopmatrix<16x4xf16, Subgroup, MatrixB>,
                                                      %c : !spirv.coopmatrix<8x4xf16, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{'spirv.KHR.CooperativeMatrixMulAdd' op Matrix Operands require all matrix element types to be Integer Types}}
  %r = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c, <AccSat> :
        !spirv.coopmatrix<8x16xf16, Subgroup, MatrixA>,
        !spirv.coopmatrix<16x4xf16, Subgroup, MatrixB> ->
          !spirv.coopmatrix<8x4xf16, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// Standard ops that can be used CooperativeMatrix types
//===----------------------------------------------------------------------===//

!matA_i32 = !spirv.coopmatrix<2x2xi32, Subgroup, MatrixA>
!matB_i32 = !spirv.coopmatrix<2x2xi32, Subgroup, MatrixB>

!matA_f32 = !spirv.coopmatrix<2x2xf32, Subgroup, MatrixA>
!matB_f32 = !spirv.coopmatrix<2x2xf32, Subgroup, MatrixB>

// These tests are kept in the same order as the list of compatible ops in the
// SPV_KHR_cooperative_matrix extension spec.

// CHECK-LABEL: @snegate
spirv.func @snegate(%a: !matA_i32, %b: !matB_i32) "None" {
  // CHECK:      spirv.SNegate {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.SNegate {{%.*}} : !spirv.coopmatrix
  %p = spirv.SNegate %a : !matA_i32
  %q = spirv.SNegate %b : !matB_i32
  spirv.Return
}

// CHECK-LABEL: @fnegate
spirv.func @fnegate(%a: !matA_f32, %b: !matB_f32) "None" {
  // CHECK:      spirv.FNegate {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.FNegate {{%.*}} : !spirv.coopmatrix
  %p = spirv.FNegate %a : !matA_f32
  %q = spirv.FNegate %b : !matB_f32
  spirv.Return
}

// CHECK-LABEL: @iadd
spirv.func @iadd(%a: !matA_i32, %b: !matB_i32) "None" {
  // CHECK:      spirv.IAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.IAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.IAdd %a, %a : !matA_i32
  %q = spirv.IAdd %b, %b : !matB_i32
  spirv.Return
}

// CHECK-LABEL: @fadd
spirv.func @fadd(%a: !matA_f32, %b: !matB_f32) "None" {
  // CHECK:      spirv.FAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.FAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.FAdd %a, %a : !matA_f32
  %q = spirv.FAdd %b, %b : !matB_f32
  spirv.Return
}

// CHECK-LABEL: @isub
spirv.func @isub(%a: !matA_i32, %b: !matB_i32) "None" {
  // CHECK:      spirv.ISub {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.ISub {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.ISub %a, %a : !matA_i32
  %q = spirv.ISub %b, %b : !matB_i32
  spirv.Return
}

// CHECK-LABEL: @fsub
spirv.func @fsub(%a: !matA_f32, %b: !matB_f32) "None" {
  // CHECK:      spirv.FSub {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.FSub {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.FSub %a, %a : !matA_f32
  %q = spirv.FSub %b, %b : !matB_f32
  spirv.Return
}

// CHECK-LABEL: @fmul
spirv.func @fmul(%a: !matA_f32, %b: !matB_f32) "None" {
  // CHECK:      spirv.FMul {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.FMul {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.FMul %a, %a : !matA_f32
  %q = spirv.FMul %b, %b : !matB_f32
  spirv.Return
}

// CHECK-LABEL: @imul
spirv.func @imul(%a: !matA_i32, %b: !matB_i32) "None" {
  // CHECK:      spirv.IMul {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.IMul {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.IMul %a, %a : !matA_i32
  %q = spirv.IMul %b, %b : !matB_i32
  spirv.Return
}

// CHECK-LABEL: @fdiv
spirv.func @fdiv(%a: !matA_f32, %b: !matB_f32) "None" {
  // CHECK:      spirv.FDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.FDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.FDiv %a, %a : !matA_f32
  %q = spirv.FDiv %b, %b : !matB_f32
  spirv.Return
}

// CHECK-LABEL: @sdiv
spirv.func @sdiv(%a: !matA_i32, %b: !matB_i32) "None" {
  // CHECK:      spirv.SDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.SDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.SDiv %a, %a : !matA_i32
  %q = spirv.SDiv %b, %b : !matB_i32
  spirv.Return
}

// CHECK-LABEL: @udiv
spirv.func @udiv(%a: !matA_i32, %b: !matB_i32) "None" {
  // CHECK:      spirv.UDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix
  // CHECK-NEXT: spirv.UDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix
  %p = spirv.UDiv %a, %a : !matA_i32
  %q = spirv.UDiv %b, %b : !matB_i32
  spirv.Return
}

// CHECK-LABEL: @matrix_times_scalar
spirv.func @matrix_times_scalar(%a: !matA_f32, %b: f32) "None" {
  // CHECK: spirv.MatrixTimesScalar {{%.*}} : !spirv.coopmatrix<2x2xf32, Subgroup, MatrixA>, f32
  %p = spirv.MatrixTimesScalar %a, %b : !matA_f32, f32
  spirv.Return
}

// -----

// For binary arithmetic instructions with coop matrix operands, the types must
// match.

spirv.func @iadd(%a: !spirv.coopmatrix<2x2xi32, Subgroup, MatrixA>,
                 %b: !spirv.coopmatrix<2x2xi32, Subgroup, MatrixB>) "None" {
  // expected-error @+1 {{op requires the same type for all operands and results}}
  %q = "spirv.IAdd"(%a, %b) :
    (!spirv.coopmatrix<2x2xi32, Subgroup, MatrixA>, !spirv.coopmatrix<2x2xi32, Subgroup, MatrixB>)
    -> !spirv.coopmatrix<2x2xi32, Subgroup, MatrixA>
  spirv.Return
}

// -----

spirv.func @fadd(%a: !spirv.coopmatrix<2x2xf32, Subgroup, MatrixA>,
                 %b: !spirv.coopmatrix<2x2xf32, Subgroup, MatrixAcc>) "None" {
  // expected-error @+1 {{op requires the same type for all operands and results}}
  %q = "spirv.FAdd"(%a, %b) :
    (!spirv.coopmatrix<2x2xf32, Subgroup, MatrixA>, !spirv.coopmatrix<2x2xf32, Subgroup, MatrixAcc>)
    -> !spirv.coopmatrix<2x2xf32, Subgroup, MatrixAcc>
  spirv.Return
}

// -----

spirv.func @matrix_times_scalar(%a: !spirv.coopmatrix<2x2xf32, Workgroup, MatrixA>, %b: f16) "None" {
  // expected-error @+1 {{input matrix components' type and scaling value must have the same type}}
  %p = spirv.MatrixTimesScalar %a, %b : !spirv.coopmatrix<2x2xf32, Workgroup, MatrixA>, f16
  spirv.Return
}
