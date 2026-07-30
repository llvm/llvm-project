; RUN: llc -mtriple=thumbv7-eabi -o - %s | FileCheck %s

declare ghccc void @g()

define ghccc void @test_direct_tail() {
; CHECK-LABEL: test_direct_tail:
; CHECK: b g

  tail call ghccc void @g()
  ret void
}

@ind_func = global ptr zeroinitializer

define ghccc void @test_indirect_tail() {
; CHECK-LABEL: test_indirect_tail:
; CHECK: bx {{r[0-9]+}}
  %func = load ptr, ptr @ind_func
  tail call ghccc void() %func()
  ret void
}
