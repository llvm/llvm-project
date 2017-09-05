; RUN: llc -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs -o - %s | FileCheck %s

; LLVM should be able to cope with multiple uses of the same flag-setting
; instruction at different points of a routine. Either by rematerializing the
; compare or by saving and restoring the flag register.

declare void @bar()

@var = global i32 0

define i32 @test_multiflag(i32 %n, i32 %m, i32 %o) {
; CHECK-LABEL: test_multiflag:

  %test = icmp ne i32 %n, %m
; CHECK: cmp [[LHS:w[0-9]+]], [[RHS:w[0-9]+]]

  %val = zext i1 %test to i32
; CHECK: cset {{[xw][0-9]+}}, ne

  store i32 %val, i32* @var

  call void @bar()
; CHECK: bl bar

  ; Currently, the comparison is emitted again. An MSR/MRS pair would also be
  ; acceptable, but assuming the call preserves NZCV is not.
  br i1 %test, label %iftrue, label %iffalse
; CHECK: cmp [[LHS]], [[RHS]]
; CHECK: b.eq

iftrue:
  ret i32 42
iffalse:
  ret i32 0
}
