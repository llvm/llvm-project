; RUN: llc -mtriple=thumbv6m-apple-macho %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-T1
; RUN: llc -mtriple=thumbv7m-apple-macho %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-T2

define i32 @test_add_i3(i1 %tst, i32 %a, i32 %b) {
; CHECK-LABEL: test_add_i3:
; CHECK: adds r0, {{r[0-9]+}}, #2
  %tmp = and i32 %a, -7
  %tmp1 = and i32 %b, -4
  %int = select i1 %tst, i32 %tmp, i32 %tmp1

  ; Call to force %int into a register that isn't r0 so using the i3 form is a
  ; good idea.
  call void @foo(i32 %int)
  %res = or i32 %int, 2
  ret i32 %res
}

define i32 @test_add_i8(i32 %a, i32 %b, i1 %tst) {
; CHECK-LABEL: test_add_i8:
; CHECK-T1: adds r0, #12
; CHECK-T2: add.w r0, {{r[0-9]+}}, #12

  %tmp = and i32 %a, -256
  %tmp1 = and i32 %b, -512
  %int = select i1 %tst, i32 %tmp, i32 %tmp1
  %res = or i32 %int, 12
  ret i32 %res
}

define i32 @test_add_i12(i32 %a, i32 %b, i1 %tst) {
; CHECK-LABEL: test_add_i12:
; CHECK-T2: addw r0, {{r[0-9]+}}, #854

  %tmp = and i32 %a, -4096
  %tmp1 = and i32 %b, -8192
  %int = select i1 %tst, i32 %tmp, i32 %tmp1
  %res = or i32 %int, 854
  ret i32 %res
}

declare void @foo(i32)
