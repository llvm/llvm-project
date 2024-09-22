; RUN: llc < %s -mtriple=i386-unknown-unknown -trap-unreachable -no-trap-after-noreturn=false -verify-machineinstrs | FileCheck %s

;; Ensure we restore caller-saved registers before EH_RETURN, even with trap-unreachable enabled.
;; For now, we deliberately avoid generating traps altogether.

; CHECK-LABEL: test32
; CHECK: pushl
; CHECK: popl
; CHECK: eh_return
; CHECK-NOT: ud2
define void @test32(i32 %offset, ptr %handler) {
  call void @llvm.eh.unwind.init()
  call void @llvm.eh.return.i32(i32 %offset, ptr %handler)
  unreachable
}
