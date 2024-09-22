; RUN: llc < %s -mtriple=x86_64-unknown-unknown -trap-unreachable -no-trap-after-noreturn=false -verify-machineinstrs | FileCheck %s

;; Ensure we restore caller-saved registers before EH_RETURN, even with trap-unreachable enabled.
;; For now, we deliberately avoid generating traps altogether.

; CHECK-LABEL: test64
; CHECK: pushq
; CHECK: popq
; CHECK: eh_return
; CHECK-NOT: ud2
define void @test64(i64 %offset, ptr %handler) {
  call void @llvm.eh.unwind.init()
  call void @llvm.eh.return.i64(i64 %offset, ptr %handler)
  unreachable
}
