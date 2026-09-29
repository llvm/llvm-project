; RUN: llc -mtriple=aarch64-win32 %s -o - | FileCheck %s

declare void @callee() noreturn

; Make sure the call isn't the last instruction in the function; if it is,
; unwinding may break.
;
; (The instruction after the call doesn't have to be anything in particular,
; but trapping has the nice side-effect of catching bugs.)

define void @test_unreachable() {
; CHECK-LABEL: test_unreachable:
; CHECK: bl      callee
; CHECK-NEXT: brk #0x1
  call void @callee() noreturn
  unreachable
}
