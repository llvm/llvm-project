; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

; Call a function jump and link with register.
define void @call_register(void (i32)* %f) {
; CHECK: jalrc $ra, $t4
; CHECK: JALRC_NM
  call void %f(i32 1)
  ret void
}
