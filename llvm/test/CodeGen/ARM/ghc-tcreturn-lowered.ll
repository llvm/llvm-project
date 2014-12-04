; RUN: llc -mtriple=thumbv7-eabi -o - %s | FileCheck %s

declare cc 10 void @g()

define cc 10 void @test_direct_tail() {
; CHECK-LABEL: test_direct_tail:
; CHECK: b g

  tail call cc10 void @g()
  ret void
}

@ind_func = global void()* zeroinitializer

define cc 10 void @test_indirect_tail() {
; CHECK-LABEL: test_indirect_tail:
; CHECK: bx {{r[0-9]+}}
  %func = load void()** @ind_func
  tail call cc10 void()* %func()
  ret void
}
