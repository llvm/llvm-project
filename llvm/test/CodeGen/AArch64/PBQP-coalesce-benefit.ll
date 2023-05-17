; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mcpu=cortex-a57 -mattr=+neon -fp-contract=fast -regalloc=pbqp -pbqp-coalescing | FileCheck %s

; CHECK-LABEL: test:
define i32 @test(i32 %acc, ptr nocapture readonly %c) {
entry:
  %0 = load i32, ptr %c, align 4
; CHECK-NOT: mov	 w{{[0-9]*}}, w0
  %add = add nsw i32 %0, %acc
  %arrayidx1 = getelementptr inbounds i32, ptr %c, i64 1
  %1 = load i32, ptr %arrayidx1, align 4
  %add2 = add nsw i32 %add, %1
  ret i32 %add2
}

