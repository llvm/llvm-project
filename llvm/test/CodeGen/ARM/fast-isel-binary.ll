; RUN: llc < %s -O0 -fast-isel-abort=1 -verify-machineinstrs -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -verify-machineinstrs -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -verify-machineinstrs -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

; Test add with non-legal types

define void @add_i1(i1 %a, i1 %b) nounwind ssp {
entry:
; ARM: add_i1
; THUMB: add_i1
  %a.addr = alloca i1, align 4
  %0 = add i1 %a, %b
; ARM: add r0, r0, r1
; THUMB: add r0, r1
  store i1 %0, ptr %a.addr, align 4
  ret void
}

define void @add_i8(i8 %a, i8 %b) nounwind ssp {
entry:
; ARM: add_i8
; THUMB: add_i8
  %a.addr = alloca i8, align 4
  %0 = add i8 %a, %b
; ARM: add r0, r0, r1
; THUMB: add r0, r1
  store i8 %0, ptr %a.addr, align 4
  ret void
}

define void @add_i16(i16 %a, i16 %b) nounwind ssp {
entry:
; ARM: add_i16
; THUMB: add_i16
  %a.addr = alloca i16, align 4
  %0 = add i16 %a, %b
; ARM: add r0, r0, r1
; THUMB: add r0, r1
  store i16 %0, ptr %a.addr, align 4
  ret void
}

; Test or with non-legal types

define void @or_i1(i1 %a, i1 %b) nounwind ssp {
entry:
; ARM: or_i1
; THUMB: or_i1
  %a.addr = alloca i1, align 4
  %0 = or i1 %a, %b
; ARM: orr r0, r0, r1
; THUMB: orrs r0, r1
  store i1 %0, ptr %a.addr, align 4
  ret void
}

define void @or_i8(i8 %a, i8 %b) nounwind ssp {
entry:
; ARM: or_i8
; THUMB: or_i8
  %a.addr = alloca i8, align 4
  %0 = or i8 %a, %b
; ARM: orr r0, r0, r1
; THUMB: orrs r0, r1
  store i8 %0, ptr %a.addr, align 4
  ret void
}

define void @or_i16(i16 %a, i16 %b) nounwind ssp {
entry:
; ARM: or_i16
; THUMB: or_i16
  %a.addr = alloca i16, align 4
  %0 = or i16 %a, %b
; ARM: orr r0, r0, r1
; THUMB: orrs r0, r1
  store i16 %0, ptr %a.addr, align 4
  ret void
}

; Test sub with non-legal types

define void @sub_i1(i1 %a, i1 %b) nounwind ssp {
entry:
; ARM: sub_i1
; THUMB: sub_i1
  %a.addr = alloca i1, align 4
  %0 = sub i1 %a, %b
; ARM: sub r0, r0, r1
; THUMB: subs r0, r0, r1
  store i1 %0, ptr %a.addr, align 4
  ret void
}

define void @sub_i8(i8 %a, i8 %b) nounwind ssp {
entry:
; ARM: sub_i8
; THUMB: sub_i8
  %a.addr = alloca i8, align 4
  %0 = sub i8 %a, %b
; ARM: sub r0, r0, r1
; THUMB: subs r0, r0, r1
  store i8 %0, ptr %a.addr, align 4
  ret void
}

define void @sub_i16(i16 %a, i16 %b) nounwind ssp {
entry:
; ARM: sub_i16
; THUMB: sub_i16
  %a.addr = alloca i16, align 4
  %0 = sub i16 %a, %b
; ARM: sub r0, r0, r1
; THUMB: subs r0, r0, r1
  store i16 %0, ptr %a.addr, align 4
  ret void
}
