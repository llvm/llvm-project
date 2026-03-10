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
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

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
  %0 = bitcast ptr %lsize2 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 @test_ndrange_2D3D.lsize2, i64 16, i1 false)
  %arraydecay = getelementptr inbounds [2 x i64], ptr %lsize2, i64 0, i64 0
  call spir_func void @_Z10ndrange_2DPKm(ptr sret(ptr) %tmp, ptr %arraydecay)
  %1 = bitcast ptr %lsize3 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %1, ptr align 8 @test_ndrange_2D3D.lsize3, i64 24, i1 false)
  %arraydecay2 = getelementptr inbounds [3 x i64], ptr %lsize3, i64 0, i64 0
  call spir_func void @_Z10ndrange_3DPKm(ptr sret(ptr) %tmp3, ptr %arraydecay2)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)

declare spir_func void @_Z10ndrange_2DPKm(ptr sret(ptr), ptr)

declare spir_func void @_Z10ndrange_3DPKm(ptr sret(ptr), ptr)

define spir_func void @test_ndrange_const_2D3D() {
entry:
  %tmp = alloca %struct.ndrange_t, align 8
  %tmp1 = alloca %struct.ndrange_t, align 8
  call spir_func void @_Z10ndrange_2DPKm(ptr sret(ptr) %tmp, ptr @test_ndrange_2D3D.lsize2)
  call spir_func void @_Z10ndrange_3DPKm(ptr sret(ptr) %tmp1, ptr @test_ndrange_2D3D.lsize3)
  ret void
}
