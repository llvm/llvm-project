; RUN: llc < %s | FileCheck %s
;
; Be sure that we ignore clobbers of unallocatable registers, rather than
; crashing.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: foo:
; CHECK: ret
define void @foo() #0 {
entry:
  call void asm sideeffect "", "~{v0}"()
  call void asm sideeffect "", "~{s0}"()
  ret void
}

attributes #0 = { nounwind "target-features"="-crypto,-fp-armv8,-neon" }
