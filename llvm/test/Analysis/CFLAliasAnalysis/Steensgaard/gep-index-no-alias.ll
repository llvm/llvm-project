; This testcase ensures that gep result does not alias gep indices

; RUN: opt < %s -disable-basicaa -cfl-steens-aa -aa-eval -print-no-aliases -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-no-aliases -disable-output 2>&1 | FileCheck %s

; CHECK: Function: foo
; CHECK: [2 x i32]* %a, [2 x i32]* %b
define void @foo(i32 %n) {
  %a = alloca [2 x i32], align 4
  %b = alloca [2 x i32], align 4
  %c = getelementptr inbounds [2 x i32], [2 x i32]* %a, i32 0, i32 %n
  %d = getelementptr inbounds [2 x i32], [2 x i32]* %b, i32 0, i32 %n
  ret void
}
