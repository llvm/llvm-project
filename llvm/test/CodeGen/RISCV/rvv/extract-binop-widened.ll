; RUN: llc -O0 -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+v -verify-machineinstrs < %s | FileCheck %s

define i64 @rotl_extract_widened() {
; CHECK-LABEL: rotl_extract_widened:
; CHECK:       lui a0, 15
; CHECK-NEXT:  addi a0, a0, -106
; CHECK-NEXT:  ret
entry:
  %vec.zero = insertelement <2 x i16> zeroinitializer, i16 0, i32 0
  %vec.const = xor <2 x i16> %vec.zero, <i16 -7273, i16 -10240>
  %vec.shifted = call <2 x i16> @llvm.fshl.v2i16(
      <2 x i16> %vec.const, <2 x i16> %vec.const,
      <2 x i16> <i16 0, i16 12>)

  %a = extractelement <2 x i16> %vec.shifted, i32 0
  %az = zext i16 %a to i64
  %ax = xor i64 %az, 1

  %b = extractelement <2 x i16> %vec.shifted, i32 1
  %bz = zext i16 %b to i64

  %r = or i64 %ax, %bz
  ret i64 %r
}

declare <2 x i16> @llvm.fshl.v2i16(<2 x i16>, <2 x i16>, <2 x i16>)
