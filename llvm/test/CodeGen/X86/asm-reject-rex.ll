; RUN: not llc -o /dev/null %s -mtriple=i386-unknown-unknown 2>&1 | FileCheck %s
; Make sure X32 still works.
; RUN: llc -o /dev/null %s -mtriple=x86_64-linux-gnux32

; CHECK: error: couldn't allocate output register for constraint '{xmm8}'
define i64 @blup() {
  %v = tail call i64 asm "", "={xmm8},0"(i64 0)
  ret i64 %v
}

; CHECK: error: couldn't allocate output register for constraint '{r8d}'
define i32 @foo() {
  %v = tail call i32 asm "", "={r8d},0"(i32 0)
  ret i32 %v
}

; CHECK: error: couldn't allocate output register for constraint '{rax}'
define i64 @bar() {
  %v = tail call i64 asm "", "={rax},0"(i64 0)
  ret i64 %v
}
