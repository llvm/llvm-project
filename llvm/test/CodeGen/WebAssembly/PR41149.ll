; RUN: llc < %s -mtriple=wasm32-unknown-unknown | FileCheck %s

; Regression test for PR41149.

define void @mod() {
; CHECK-LABEL: mod:
; CHECK-NEXT: .functype mod () -> ()
; CHECK:      local.get       0
; CHECK-NEXT: local.get       0
; CHECK-NEXT: i32.load8_s     0
; CHECK-NEXT: local.tee       0
; CHECK-NEXT: local.get       0
; CHECK-NEXT: i32.const       31
; CHECK-NEXT: i32.shr_s
; CHECK-NEXT: local.tee       0
; CHECK-NEXT: i32.add
; CHECK-NEXT: local.get       0
; CHECK-NEXT: i32.xor
; CHECK-NEXT: i32.store8      0
  %tmp = load <4 x i8>, <4 x i8>* undef
  %tmp2 = icmp slt <4 x i8> %tmp, zeroinitializer
  %tmp3 = sub <4 x i8> zeroinitializer, %tmp
  %tmp4 = select <4 x i1> %tmp2, <4 x i8> %tmp3, <4 x i8> %tmp
  store <4 x i8> %tmp4, <4 x i8>* undef
  ret void
}
