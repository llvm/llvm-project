; RUN: llc -mtriple=i686-unknown-linux-gnu -o - %s | FileCheck %s

declare void @g_bool(i1 zeroext)
define void @f_bool(i1 zeroext %x) {
entry:
  tail call void @g_bool(i1 zeroext %x)
  ret void

; Forwarding a bool in a tail call works.
; CHECK-LABEL: f_bool:
; CHECK-NOT:   movz
; CHECK:       jmp g_bool
}


declare void @g_float(float)
define void @f_i32(i32 %x) {
entry:
  %0 = bitcast i32 %x to float
  tail call void @g_float(float %0)
  ret void

; Forwarding a bitcasted value works too.
; CHECK-LABEL: f_i32
; CHECK-NOT:   mov
; CHECK:       jmp g_float
}
