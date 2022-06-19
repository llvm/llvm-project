; RUN: llc < %s -march=nvptx -mcpu=sm_35 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_35 | %ptxas-verify -arch=sm_35 %}

; Verify that we correctly emit code for extending ldg/ldu. We do not expose
; extending variants in the backend, but the ldg/ldu selection code may pick
; extending loads as candidates. We do want to support this, so make sure we
; emit the necessary cvt.* instructions to implement the extension and let ptxas
; emit the real extending loads.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: spam
define ptx_kernel void @spam(i8 addrspace(1)* noalias nocapture readonly %arg, i8 addrspace(1)* noalias nocapture %arg1, i64 %arg2, i64 %arg3) #0 {
bb:
  %tmp = bitcast i8 addrspace(1)* %arg to i16 addrspace(1)*
  %tmp4 = bitcast i8 addrspace(1)* %arg1 to i64 addrspace(1)*
  %tmp5 = add nsw i64 %arg3, 8
  %tmp6 = getelementptr i16, i16 addrspace(1)* %tmp, i64 %tmp5
; CHECK: ld.global.nc.u16
  %tmp7 = load i16, i16 addrspace(1)* %tmp6, align 2
; CHECK: cvt.s32.s16
  %tmp8 = sext i16 %tmp7 to i64
  %tmp9 = mul nsw i64 %tmp8, %tmp8
  %tmp10 = load i64, i64 addrspace(1)* %tmp4, align 8
  %tmp11 = add nsw i64 %tmp9, %tmp10
  store i64 %tmp11, i64 addrspace(1)* %tmp4, align 8
  ret void
}

attributes #0 = { norecurse nounwind "polly.skip.fn" }

!nvvm.annotations = !{!0}

!0 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i64, i64)* @spam, !"maxntidx", i64 1, !"maxntidy", i64 1, !"maxntidz", i64 1}
