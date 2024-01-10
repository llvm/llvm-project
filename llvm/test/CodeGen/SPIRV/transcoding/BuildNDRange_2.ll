;; Test what ndrange_2D and ndrange_3D can coexist in the same module
;;
;; bash$ cat BuildNDRange_2.cl
;; void test_ndrange_2D3D() {
;;   size_t lsize2[2] = {1, 1};
;;   ndrange_2D(lsize2);
;;
;;   size_t lsize3[3] = {1, 1, 1};
;;   ndrange_3D(lsize3);
;; }
;;
;; void test_ndrange_const_2D3D() {
;;   const size_t lsize2[2] = {1, 1};
;;   ndrange_2D(lsize2);
;;
;;   const size_t lsize3[3] = {1, 1, 1};
;;   ndrange_3D(lsize3);
;; }
;; bash$ $PATH_TO_GEN/bin/clang -cc1 -x cl -cl-std=CL2.0 -triple spir64-unknown-unknown -emit-llvm  -include opencl-20.h  BuildNDRange_2.cl -o BuildNDRange_2.ll

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *
; REQUIRES: asserts

; CHECK-SPIRV-DAG:     %[[#LEN2_ID:]] = OpConstant %[[#]] 2
; CHECK-SPIRV-DAG:     %[[#LEN3_ID:]] = OpConstant %[[#]] 3
; CHECK-SPIRV-DAG:     %[[#ARRAY_T2:]] = OpTypeArray %[[#]] %[[#LEN2_ID]]
; CHECK-SPIRV-DAG:     %[[#ARRAY_T3:]] = OpTypeArray %[[#]] %[[#LEN3_ID]]

; CHECK-SPIRV-LABEL:   OpFunction
; CHECK-SPIRV:         %[[#LOAD2_ID:]] = OpLoad %[[#ARRAY_T2]]
; CHECK-SPIRV:         %[[#]] = OpBuildNDRange %[[#]] %[[#LOAD2_ID]]
; CHECK-SPIRV:         %[[#LOAD3_ID:]] = OpLoad %[[#ARRAY_T3]]
; CHECK-SPIRV:         %[[#]] = OpBuildNDRange %[[#]] %[[#LOAD3_ID]]
; CHECK-SPIRV-LABEL:   OpFunctionEnd

; CHECK-SPIRV-LABEL:   OpFunction
; CHECK-SPIRV:         %[[#CONST_LOAD2_ID:]] = OpLoad %[[#ARRAY_T2]]
; CHECK-SPIRV:         %[[#]] = OpBuildNDRange %[[#]] %[[#CONST_LOAD2_ID]]
; CHECK-SPIRV:         %[[#CONST_LOAD3_ID:]] = OpLoad %[[#ARRAY_T3]]
; CHECK-SPIRV:         %[[#]] = OpBuildNDRange %[[#]] %[[#CONST_LOAD3_ID]]
; CHECK-SPIRV-LABEL:   OpFunctionEnd

%struct.ndrange_t = type { i32, [3 x i64], [3 x i64], [3 x i64] }

@test_ndrange_2D3D.lsize2 = private constant [2 x i64] [i64 1, i64 1], align 8
@test_ndrange_2D3D.lsize3 = private constant [3 x i64] [i64 1, i64 1, i64 1], align 8


define spir_func void @test_ndrange_2D3D() {
entry:
  %lsize2 = alloca [2 x i64], align 8
  %tmp = alloca %struct.ndrange_t, align 8
  %lsize3 = alloca [3 x i64], align 8
  %tmp3 = alloca %struct.ndrange_t, align 8
  %0 = bitcast [2 x i64]* %lsize2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 bitcast ([2 x i64]* @test_ndrange_2D3D.lsize2 to i8*), i64 16, i1 false)
  %arraydecay = getelementptr inbounds [2 x i64], [2 x i64]* %lsize2, i64 0, i64 0
  call spir_func void @_Z10ndrange_2DPKm(%struct.ndrange_t* sret(%struct.ndrange_t*) %tmp, i64* %arraydecay)
  %1 = bitcast [3 x i64]* %lsize3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %1, i8* align 8 bitcast ([3 x i64]* @test_ndrange_2D3D.lsize3 to i8*), i64 24, i1 false)
  %arraydecay2 = getelementptr inbounds [3 x i64], [3 x i64]* %lsize3, i64 0, i64 0
  call spir_func void @_Z10ndrange_3DPKm(%struct.ndrange_t* sret(%struct.ndrange_t*) %tmp3, i64* %arraydecay2)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)

declare spir_func void @_Z10ndrange_2DPKm(%struct.ndrange_t* sret(%struct.ndrange_t*), i64*)

declare spir_func void @_Z10ndrange_3DPKm(%struct.ndrange_t* sret(%struct.ndrange_t*), i64*)

define spir_func void @test_ndrange_const_2D3D() {
entry:
  %tmp = alloca %struct.ndrange_t, align 8
  %tmp1 = alloca %struct.ndrange_t, align 8
  call spir_func void @_Z10ndrange_2DPKm(%struct.ndrange_t* sret(%struct.ndrange_t*) %tmp, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @test_ndrange_2D3D.lsize2, i64 0, i64 0))
  call spir_func void @_Z10ndrange_3DPKm(%struct.ndrange_t* sret(%struct.ndrange_t*) %tmp1, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @test_ndrange_2D3D.lsize3, i64 0, i64 0))
  ret void
}
