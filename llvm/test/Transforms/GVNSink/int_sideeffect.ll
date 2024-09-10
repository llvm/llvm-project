; RUN: opt -S < %s -passes=gvn-sink | FileCheck %s

declare void @llvm.sideeffect()

; GVN sinking across a @llvm.sideeffect.

; CHECK-LABEL: scalarsSinking
; CHECK-NOT: fmul
; CHECK: = phi
; CHECK: = fmul
define float @scalarsSinking(float %d, float %m, float %a, i1 %cmp) {
entry:
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @llvm.sideeffect()
  %sub = fsub float %m, %a
  %mul0 = fmul float %sub, %d
  br label %if.end

if.else:
  %add = fadd float %m, %a
  %mul1 = fmul float %add, %d
  br label %if.end

if.end:
  %phi = phi float [ %mul0, %if.then ], [ %mul1, %if.else ]
  ret float %phi
}

; CHECK-LABEL: scalarsSinkingReverse
; CHECK-NOT: fmul
; CHECK: = phi
; CHECK: = fmul
define float @scalarsSinkingReverse(float %d, float %m, float %a, i1 %cmp) {
; This test is just a reverse(graph mirror) of the test
; above to ensure GVNSink doesn't depend on the order of branches.
entry:
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %add = fadd float %m, %a
  %mul1 = fmul float %add, %d
  br label %if.end

if.else:
  call void @llvm.sideeffect()
  %sub = fsub float %m, %a
  %mul0 = fmul float %sub, %d
  br label %if.end

if.end:
  %phi = phi float [ %mul1, %if.then ], [ %mul0, %if.else ]
  ret float %phi
}

