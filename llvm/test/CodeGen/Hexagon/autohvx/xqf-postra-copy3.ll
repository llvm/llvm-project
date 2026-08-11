; Tests if the 1st operand of vasr instruction is converted to qf
; It should NOT be.

; RUN: llc -mtriple=hexagon -O0 -mv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat \
; RUN: -enable-xqf-gen=true -hexagon-qfloat-mode=lossy < %s -o - | FileCheck %s

; CHECK-LABEL: main:
; CHECK: v[[SH:[0-9]+]] = vxor(v[[SH]],v[[SH]])
; CHECK-NOT: qf16
; CHECK-NOT: qf32
; CHECK: v{{[0-9]+}}.ub = vasr(v{{[0-9]+:[0-9]+}}.uh,v[[SH]].ub):rnd:sat

define i32 @main() {
entry:
  %0 = call <32 x i32> @llvm.hexagon.V6.vasrvuhubrndsat.128B(<64 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <32 x i32> zeroinitializer)
  store <32 x i32> %0, ptr null, align 128
  ret i32 0
}

declare <32 x i32> @llvm.hexagon.V6.vasrvuhubrndsat.128B(<64 x i32>, <32 x i32>)
