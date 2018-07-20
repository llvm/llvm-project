; Bugzilla: https://bugs.llvm.org/show_bug.cgi?id=33623
; RUN: llvm-diff %s %s

define i32 @foo(i8*) {
entry:
  indirectbr i8* %0, [label %A, label %B, label %entry]
A:
  ret i32 1
B:
  ret i32 2
}
