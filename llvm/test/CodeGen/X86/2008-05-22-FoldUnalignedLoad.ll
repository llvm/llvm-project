; RUN: llc < %s -march=x86 -mcpu=penryn | FileCheck %s

define void @a(<4 x float>* %x) nounwind  {
entry:
        %tmp2 = load <4 x float>, <4 x float>* %x, align 1
        %inv = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %tmp2)
        store <4 x float> %inv, <4 x float>* %x, align 1
        ret void
}

; CHECK-LABEL: a:
; CHECK: movups
; CHECK: movups
; CHECK-NOT: movups
; CHECK: ret

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>)
