; RUN: llc -verify-machineinstrs < %s -mcpu=ppc32 -mattr=+crbits | FileCheck %s
target triple = "powerpc-unknown-linux-gnu"

define void @check_callee(
  i32, i32, i32, i32,
  i32, i32, i32, i32,
  i1 zeroext %s1
) {
  call void @check_caller(
    i32 9, i32 9, i32 9, i32 9,
    i32 9, i32 9, i32 9, i32 9,
    i1 zeroext %s1)
  ret void
}

; CHECK-LABEL: @check_callee
; CHECK: lbz {{[0-9]+}}, 27(1)
; CHECK: stw {{[0-9]+}}, 8(1)

declare void @check_caller(
  i32, i32, i32, i32,
  i32, i32, i32, i32,
  i1 zeroext
)
