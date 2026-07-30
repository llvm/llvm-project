; RUN: llc < %s -O0 -fast-isel-abort=1 -verify-machineinstrs -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

; Target-specific selector can't properly handle the double because it isn't
; being passed via a register, so the materialized arguments become dead code.

define i32 @main(i32 %argc, ptr %argv) nounwind {
entry:
; THUMB: main
  call void @printArgsNoRet(i32 1, float 0x4000CCCCC0000000, i8 signext 99, double 4.100000e+00)
; THUMB: bl _printArgsNoRet
; THUMB-NOT: ldr
; THUMB-NOT: vldr
; THUMB-NOT: vmov
; THUMB-NOT: ldr
; THUMB-NOT: sxtb
; THUMB: movs r0, #0
; THUMB: pop
  ret i32 0
}

declare void @printArgsNoRet(i32 %a1, float %a2, i8 signext %a3, double %a4)
