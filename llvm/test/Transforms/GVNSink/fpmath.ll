; RUN: opt -S < %s -gvn-sink | FileCheck %s

; CHECK-LABEL: sink
; CHECK-NOT: fneg
; CHECK: = phi
; CHECK: fneg
define float @sink(float %a, float %b, i1 %cmp) {
entry:
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %sub = fsub float %a, %b
  %neg0 = fneg float %sub
  br label %if.end

if.else:
  %add = fadd float %a, %b
  %neg1 = fneg float %add
  br label %if.end

if.end:
  %phi = phi float [ %neg0, %if.then ], [ %neg1, %if.else ]
  ret float %phi
}
