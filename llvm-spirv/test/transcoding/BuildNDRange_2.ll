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
;; bash$ $PATH_TO_GEN/bin/clang -cc1 -x cl -cl-std=CL2.0 -triple spir64-unknonw-unknown -emit-llvm  -include opencl-20.h  BuildNDRange_2.cl -o BuildNDRange_2.ll

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG:     Constant {{[0-9]+}} [[LEN2_ID:[0-9]+]] 2
; CHECK-SPIRV-DAG:     Constant {{[0-9]+}} [[LEN3_ID:[0-9]+]] 3
; CHECK-SPIRV-DAG:     TypeArray [[ARRAY_T2:[0-9]+]] {{[0-9]+}} [[LEN2_ID]]
; CHECK-SPIRV-DAG:     TypeArray [[ARRAY_T3:[0-9]+]] {{[0-9]+}} [[LEN3_ID]]

; CHECK-SPIRV-LABEL:   5 Function
; CHECK-SPIRV:         Load [[ARRAY_T2]] [[LOAD2_ID:[0-9]+]]
; CHECK-SPIRV:         BuildNDRange {{[0-9]+}} {{[0-9]+}} [[LOAD2_ID]]
; CHECK-SPIRV:         Load [[ARRAY_T3]] [[LOAD3_ID:[0-9]+]]
; CHECK-SPIRV:         BuildNDRange {{[0-9]+}} {{[0-9]+}} [[LOAD3_ID]]
; CHECK-SPIRV-LABEL:   1 FunctionEnd

; CHECK-SPIRV-LABEL:   5 Function
; CHECK-SPIRV:         Load [[ARRAY_T2]] [[CONST_LOAD2_ID:[0-9]+]]
; CHECK-SPIRV:         BuildNDRange {{[0-9]+}} {{[0-9]+}} [[CONST_LOAD2_ID]]
; CHECK-SPIRV:         Load [[ARRAY_T3]] [[CONST_LOAD3_ID:[0-9]+]]
; CHECK-SPIRV:         BuildNDRange {{[0-9]+}} {{[0-9]+}} [[CONST_LOAD3_ID]]
; CHECK-SPIRV-LABEL:   1 FunctionEnd

; CHECK-LLVM-LABEL: @test_ndrange_2D3D
; CHECK-LLVM:       call spir_func void @_Z10ndrange_2D
; CHECK-LLVM:       call spir_func void @_Z10ndrange_3D
; CHECK-LLVM-LABEL: @test_ndrange_const_2D3D
; CHECK-LLVM:       call spir_func void @_Z10ndrange_2D
; CHECK-LLVM:       call spir_func void @_Z10ndrange_3D

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknonw-unknown"

%struct.ndrange_t = type { i32, [3 x i64], [3 x i64], [3 x i64] }

@test_ndrange_2D3D.lsize2 = private constant [2 x i64] [i64 1, i64 1], align 8
@test_ndrange_2D3D.lsize3 = private constant [3 x i64] [i64 1, i64 1, i64 1], align 8


; Function Attrs: nounwind
define spir_func void @test_ndrange_2D3D() #0 {
entry:
  %lsize2 = alloca [2 x i64], align 8
  %tmp = alloca %struct.ndrange_t, align 8
  %lsize3 = alloca [3 x i64], align 8
  %tmp3 = alloca %struct.ndrange_t, align 8
  %0 = bitcast [2 x i64]* %lsize2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ([2 x i64]* @test_ndrange_2D3D.lsize2 to i8*), i64 16, i32 8, i1 false)
  %arraydecay = getelementptr inbounds [2 x i64], [2 x i64]* %lsize2, i64 0, i64 0
  call spir_func void @_Z10ndrange_2DPKm(%struct.ndrange_t* sret %tmp, i64* %arraydecay) #2
  %1 = bitcast [3 x i64]* %lsize3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast ([3 x i64]* @test_ndrange_2D3D.lsize3 to i8*), i64 24, i32 8, i1 false)
  %arraydecay2 = getelementptr inbounds [3 x i64], [3 x i64]* %lsize3, i64 0, i64 0
  call spir_func void @_Z10ndrange_3DPKm(%struct.ndrange_t* sret %tmp3, i64* %arraydecay2) #2
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

declare spir_func void @_Z10ndrange_2DPKm(%struct.ndrange_t* sret, i64*) #1

declare spir_func void @_Z10ndrange_3DPKm(%struct.ndrange_t* sret, i64*) #1

; Function Attrs: nounwind
define spir_func void @test_ndrange_const_2D3D() #0 {
entry:
  %tmp = alloca %struct.ndrange_t, align 8
  %tmp1 = alloca %struct.ndrange_t, align 8
  call spir_func void @_Z10ndrange_2DPKm(%struct.ndrange_t* sret %tmp, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @test_ndrange_2D3D.lsize2, i64 0, i64 0)) #2
  call spir_func void @_Z10ndrange_3DPKm(%struct.ndrange_t* sret %tmp1, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @test_ndrange_2D3D.lsize3, i64 0, i64 0)) #2
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}

!0 = !{i32 2, i32 0}
!1 = !{i32 2, i32 0}
!2 = !{}
