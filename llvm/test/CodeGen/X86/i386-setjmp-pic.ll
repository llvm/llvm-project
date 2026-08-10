; RUN: llc  -verify-machineinstrs -relocation-model=pic %s -o - | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx"

; Check that the register used as base pointer for setjmp
; is properly initialized.
; The test used to fail with the machine verifier complaining
; that the global base pointer is not initialized.
; PR26742.
;
; CHECK: test:
; CHECK: calll [[BP_SETUP_LABEL:L[$0-9a-zA-Z_-]+]]
; CHECK: [[BP_SETUP_LABEL]]:
; CHECK-NEXT: popl [[BP:%[a-z]+]]
;
; CHECK: leal [[BLOCK_ADDR:LBB[$0-9a-zA-Z_-]+]]-[[BP_SETUP_LABEL]]([[BP]]),
define i32 @test(ptr %tmp) {
entry:
  %tmp9 = call i32 @llvm.eh.sjlj.setjmp(ptr %tmp)
  ret i32 %tmp9
}

declare i32 @llvm.eh.sjlj.setjmp(ptr)
