; RUN: not opt < %s -passes=fix-irreducible -S 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: unsupported block terminator: fix-irreducible only supports br and callbr instructions

define void @loop_1(i32 %Value, i1 %PredEntry) {
entry:
  br i1 %PredEntry, label %A, label %B

A:
  br label %B

B:
  switch i32 %Value, label %exit [
    i32 0, label %A
    i32 1, label %B
  ]

exit:
  ret void
}
