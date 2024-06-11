; Adapted from https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/main/test/extensions/INTEL/SPV_INTEL_cache_controls

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_cache_controls %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_cache_controls %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: Capability CacheControlsINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_cache_controls"
; CHECK-SPIRV-DAG: OpName %[[#GVar:]] "G"
; CHECK-SPIRV-DAG: OpName %[[#Arg:]] "buffer"
; CHECK-SPIRV-DAG: OpDecorate %[[#GVar]] CacheControlStoreINTEL 0 1
; CHECK-SPIRV-DAG: OpDecorate %[[#GVar]] CacheControlStoreINTEL 1 3
; CHECK-SPIRV-DAG: OpDecorate %[[#Arg]] CacheControlLoadINTEL 0 0
; CHECK-SPIRV-DAG: OpDecorate %[[#Arg]] CacheControlStoreINTEL 0 1
; CHECK-SPIRV-DAG: OpDecorate %[[#LoadPtr:]] CacheControlLoadINTEL 0 1
; CHECK-SPIRV-DAG: OpDecorate %[[#LoadPtr]] CacheControlLoadINTEL 1 1
; CHECK-SPIRV-DAG: OpDecorate %[[#StorePtr:]] CacheControlStoreINTEL 0 1
; CHECK-SPIRV-DAG: OpDecorate %[[#StorePtr]] CacheControlStoreINTEL 1 2
; CHECK-SPIRV: OpLoad %[[#]] %[[#LoadPtr]]
; CHECK-SPIRV: OpStore %[[#StorePtr]] %[[#]]

@G = common addrspace(1) global i32 0, align 4, !spirv.Decorations !9

define spir_kernel void @test(ptr addrspace(1) %dummy, ptr addrspace(1) %buffer) !spirv.ParameterDecorations !12 {
entry:
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %buffer, i64 1, !spirv.Decorations !3
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %buffer, i64 0, !spirv.Decorations !6
  store i32 %0, ptr addrspace(1) %arrayidx1, align 4
  ret void
}

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{i32 1, i32 2}
!3 = !{!4, !5}
!4 = !{i32 6442, i32 0, i32 1}  ; {CacheControlLoadINTEL, CacheLevel=0, Cached}
!5 = !{i32 6442, i32 1, i32 1}  ; {CacheControlLoadINTEL, CacheLevel=1, Cached}
!6 = !{!7, !8}
!7 = !{i32 6443, i32 0, i32 1}  ; {CacheControlStoreINTEL, CacheLevel=0, WriteThrough}
!8 = !{i32 6443, i32 1, i32 2}  ; {CacheControlStoreINTEL, CacheLevel=1, WriteBack}
!9 = !{!10, !11}
!10 = !{i32 6443, i32 0, i32 1}  ; {CacheControlStoreINTEL, CacheLevel=0, WriteThrough}
!11 = !{i32 6443, i32 1, i32 3}  ; {CacheControlStoreINTEL, CacheLevel=1, Streaming}
!12 = !{!13, !14}
!13 = !{}
!14 = !{!15, !16}
!15 = !{i32 6442, i32 0, i32 0}  ; {CacheControlLoadINTEL,   CacheLevel=0, Uncached}
!16 = !{i32 6443, i32 0, i32 1}  ; {CacheControlStoreINTEL,  CacheLevel=0, WriteThrough}
