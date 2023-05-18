; This test verifies that no optimizations are performed on the @f function
; when the -opt-bisect-limit=0 option is used.  In particular, the X86
; instruction selector will optimize the cmp instruction to a sub instruction
; if it is not run in -O0 mode.

; RUN: llc -O3 -opt-bisect-limit=0 -o - %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @f() {
entry:
  %cmp = icmp slt i32 undef, 8
  br i1 %cmp, label %middle, label %end

middle:
  br label %end

end:
  ret void
}

; CHECK: cmpl $8, %eax
