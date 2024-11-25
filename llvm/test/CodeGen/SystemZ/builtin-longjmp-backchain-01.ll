;Test -mbackchain longjmp load from jmp_buf.
; Frame pointer from Slot 1.
; Jump address from Slot 2.
; Backchain Value from Slot 3.
; Stack Pointer from Slot 4.
; Literal Pool Pointer from Slot 5.

; RUN: llc -O2 < %s | FileCheck %s

@buf = dso_local global [20 x ptr] zeroinitializer, align 8

; Function Attrs: noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
; CHECK:        stmg    %r11, %r15, 88(%r15)
; CHECK:        larl    %r1, buf
; CHECK:        lg      %r2, 8(%r1)
; CHECK:        lg      %r11, 0(%r1)
; CHECK:        lg      %r13, 32(%r1)
; CHECK:        lg      %r3, 16(%r1)
; CHECK:        lg      %r15, 24(%r1)
; CHECK:        stg     %r3, 0(%r15)
; CHECK:        br      %r2

  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #1

attributes #0 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { noreturn nounwind }

