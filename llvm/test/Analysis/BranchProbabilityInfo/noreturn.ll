; Test the static branch probability heuristics for no-return functions.
; RUN: opt < %s -analyze -branch-prob | FileCheck %s

declare void @abort() noreturn

define i32 @test1(i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test1'
entry:
  %cond = icmp eq i32 %a, 42
  br i1 %cond, label %exit, label %abort
; CHECK: edge entry -> exit probability is 1048575 / 1048576
; CHECK: edge entry -> abort probability is 1 / 1048576

abort:
  call void @abort() noreturn
  unreachable

exit:
  ret i32 %b
}

define i32 @test2(i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test2'
entry:
  switch i32 %a, label %exit [i32 1, label %case_a
                              i32 2, label %case_b
                              i32 3, label %case_c
                              i32 4, label %case_d]
; CHECK: edge entry -> exit probability is 1048575 / 1048579
; CHECK: edge entry -> case_a probability is 1 / 1048579
; CHECK: edge entry -> case_b probability is 1 / 1048579
; CHECK: edge entry -> case_c probability is 1 / 1048579
; CHECK: edge entry -> case_d probability is 1 / 1048579

case_a:
  br label %case_b

case_b:
  br label %case_c

case_c:
  br label %case_d

case_d:
  call void @abort() noreturn
  unreachable

exit:
  ret i32 %b
}

define i32 @test3(i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test3'
; Make sure we unify across multiple conditional branches.
entry:
  %cond1 = icmp eq i32 %a, 42
  br i1 %cond1, label %exit, label %dom
; CHECK: edge entry -> exit probability is 1048575 / 1048576
; CHECK: edge entry -> dom probability is 1 / 1048576

dom:
  %cond2 = icmp ult i32 %a, 42
  br i1 %cond2, label %idom1, label %idom2
; CHECK: edge dom -> idom1 probability is 1 / 2
; CHECK: edge dom -> idom2 probability is 1 / 2

idom1:
  br label %abort

idom2:
  br label %abort

abort:
  call void @abort() noreturn
  unreachable

exit:
  ret i32 %b
}
