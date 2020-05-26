; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

; Test add with non-legal types

define void @add_i8(i8 %a, i8 %b) nounwind {
entry:
; ELF64: add_i8
  %a.addr = alloca i8, align 4
  %0 = add i8 %a, %b
; ELF64: add
  store i8 %0, i8* %a.addr, align 4
  ret void
}

define void @add_i8_imm(i8 %a) nounwind {
entry:
; ELF64: add_i8_imm
  %a.addr = alloca i8, align 4
  %0 = add i8 %a, 22;
; ELF64: addi
  store i8 %0, i8* %a.addr, align 4
  ret void
}

define void @add_i16(i16 %a, i16 %b) nounwind {
entry:
; ELF64: add_i16
  %a.addr = alloca i16, align 4
  %0 = add i16 %a, %b
; ELF64: add
  store i16 %0, i16* %a.addr, align 4
  ret void
}

define void @add_i16_imm(i16 %a, i16 %b) nounwind {
entry:
; ELF64: add_i16_imm
  %a.addr = alloca i16, align 4
  %0 = add i16 %a, 243;
; ELF64: addi
  store i16 %0, i16* %a.addr, align 4
  ret void
}

; Test or with non-legal types

define void @or_i8(i8 %a, i8 %b) nounwind {
entry:
; ELF64: or_i8
  %a.addr = alloca i8, align 4
  %0 = or i8 %a, %b
; ELF64: or
  store i8 %0, i8* %a.addr, align 4
  ret void
}

define void @or_i8_imm(i8 %a) nounwind {
entry:
; ELF64: or_i8_imm
  %a.addr = alloca i8, align 4
  %0 = or i8 %a, -13;
; ELF64: ori
  store i8 %0, i8* %a.addr, align 4
  ret void
}

define void @or_i16(i16 %a, i16 %b) nounwind {
entry:
; ELF64: or_i16
  %a.addr = alloca i16, align 4
  %0 = or i16 %a, %b
; ELF64: or
  store i16 %0, i16* %a.addr, align 4
  ret void
}

define void @or_i16_imm(i16 %a) nounwind {
entry:
; ELF64: or_i16_imm
  %a.addr = alloca i16, align 4
  %0 = or i16 %a, 273;
; ELF64: ori
  store i16 %0, i16* %a.addr, align 4
  ret void
}

; Test sub with non-legal types

define void @sub_i8(i8 %a, i8 %b) nounwind {
entry:
; ELF64: sub_i8
  %a.addr = alloca i8, align 4
  %0 = sub i8 %a, %b
; ELF64: sub
  store i8 %0, i8* %a.addr, align 4
  ret void
}

define void @sub_i8_imm(i8 %a) nounwind {
entry:
; ELF64: sub_i8_imm
  %a.addr = alloca i8, align 4
  %0 = sub i8 %a, 22;
; ELF64: addi
  store i8 %0, i8* %a.addr, align 4
  ret void
}

define void @sub_i16(i16 %a, i16 %b) nounwind {
entry:
; ELF64: sub_i16
  %a.addr = alloca i16, align 4
  %0 = sub i16 %a, %b
; ELF64: sub
  store i16 %0, i16* %a.addr, align 4
  ret void
}

define void @sub_i16_imm(i16 %a) nounwind {
entry:
; ELF64: sub_i16_imm
  %a.addr = alloca i16, align 4
  %0 = sub i16 %a, 247;
; ELF64: addi
  store i16 %0, i16* %a.addr, align 4
  ret void
}

define void @sub_i16_badimm(i16 %a) nounwind {
entry:
; ELF64: sub_i16_imm
  %a.addr = alloca i16, align 4
  %0 = sub i16 %a, -32768;
; ELF64: sub
  store i16 %0, i16* %a.addr, align 4
  ret void
}
