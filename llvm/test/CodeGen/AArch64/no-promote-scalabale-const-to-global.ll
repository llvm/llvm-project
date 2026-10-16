; RUN: llc -mtriple=aarch64 -mattr=+sve -stop-after=aarch64-promote-const < %s | FileCheck %s

; Test that the constant inside the `phi` is not promoted to a global
; CHECK-NOT: _PromotedConst
define void @f(i1 %c, ptr %p, ptr %q) {
entry:
  br i1 %c, label %if.then, label %if.else

if.then:
  %u = load [2 x <vscale x 4 x float> ], ptr %p
  br label %exit

if.else:
  br label %exit

exit:
  %v = phi [2 x <vscale x 4 x float> ]  [ %u, %if.then], [[<vscale x 4 x float> zeroinitializer, <vscale x 4 x float> poison], %if.else]
  store [2 x <vscale x 4 x float>] %v, ptr %q
  ret void
}
