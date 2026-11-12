; RUN: opt -S < %s -passes=gvn-hoist | FileCheck %s

declare void @llvm.sideeffect()

; GVN hoisting across a @llvm.sideeffect.

; CHECK-LABEL: scalarsHoisting
; CHECK: = fsub
; CHECK: br i1 %cmp,
; CHECK-NOT: fsub
define float @scalarsHoisting(float %d, float %m, float %a, i1 %cmp) {
entry:
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @llvm.sideeffect()
  %sub0 = fsub float %m, %a
  %mul = fmul float %sub0, %d
  br label %if.end

if.else:
  %sub1 = fsub float %m, %a
  %div = fdiv float %sub1, %d
  br label %if.end

if.end:
  %phi = phi float [ %mul, %if.then ], [ %div, %if.else ]
  ret float %phi
}

