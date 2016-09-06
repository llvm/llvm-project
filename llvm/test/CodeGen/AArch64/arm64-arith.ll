; RUN: llc < %s -mtriple=arm64-eabi -asm-verbose=false | FileCheck %s

define i32 @t1(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t1:
; CHECK: add w0, w1, w0
; CHECK: ret
  %add = add i32 %b, %a
  ret i32 %add
}

define i32 @t2(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t2:
; CHECK: udiv w0, w0, w1
; CHECK: ret
  %udiv = udiv i32 %a, %b
  ret i32 %udiv
}

define i64 @t3(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t3:
; CHECK: udiv x0, x0, x1
; CHECK: ret
  %udiv = udiv i64 %a, %b
  ret i64 %udiv
}

define i32 @t4(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t4:
; CHECK: sdiv w0, w0, w1
; CHECK: ret
  %sdiv = sdiv i32 %a, %b
  ret i32 %sdiv
}

define i64 @t5(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t5:
; CHECK: sdiv x0, x0, x1
; CHECK: ret
  %sdiv = sdiv i64 %a, %b
  ret i64 %sdiv
}

define i32 @t6(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t6:
; CHECK: lsl w0, w0, w1
; CHECK: ret
  %shl = shl i32 %a, %b
  ret i32 %shl
}

define i64 @t7(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t7:
; CHECK: lsl x0, x0, x1
; CHECK: ret
  %shl = shl i64 %a, %b
  ret i64 %shl
}

define i32 @t8(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t8:
; CHECK: lsr w0, w0, w1
; CHECK: ret
  %lshr = lshr i32 %a, %b
  ret i32 %lshr
}

define i64 @t9(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t9:
; CHECK: lsr x0, x0, x1
; CHECK: ret
  %lshr = lshr i64 %a, %b
  ret i64 %lshr
}

define i32 @t10(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t10:
; CHECK: asr w0, w0, w1
; CHECK: ret
  %ashr = ashr i32 %a, %b
  ret i32 %ashr
}

define i64 @t11(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t11:
; CHECK: asr x0, x0, x1
; CHECK: ret
  %ashr = ashr i64 %a, %b
  ret i64 %ashr
}

define i32 @t12(i16 %a, i32 %x) nounwind ssp {
entry:
; CHECK-LABEL: t12:
; CHECK: add	w0, w1, w0, sxth
; CHECK: ret
  %c = sext i16 %a to i32
  %e = add i32 %x, %c
  ret i32 %e
}

define i32 @t13(i16 %a, i32 %x) nounwind ssp {
entry:
; CHECK-LABEL: t13:
; CHECK: add	w0, w1, w0, sxth #2
; CHECK: ret
  %c = sext i16 %a to i32
  %d = shl i32 %c, 2
  %e = add i32 %x, %d
  ret i32 %e
}

define i64 @t14(i16 %a, i64 %x) nounwind ssp {
entry:
; CHECK-LABEL: t14:
; CHECK: and	w8, w0, #0xffff
; CHECK: add	x0, x1, w8, uxtw #3
; CHECK: ret
  %c = zext i16 %a to i64
  %d = shl i64 %c, 3
  %e = add i64 %x, %d
  ret i64 %e
}

; rdar://9160598
define i64 @t15(i64 %a, i64 %x) nounwind ssp {
entry:
; CHECK-LABEL: t15:
; CHECK: add x0, x1, w0, uxtw
; CHECK: ret
  %b = and i64 %a, 4294967295
  %c = add i64 %x, %b
  ret i64 %c
}

define i64 @t16(i64 %x) nounwind ssp {
entry:
; CHECK-LABEL: t16:
; CHECK: lsl x0, x0, #1
; CHECK: ret
  %a = shl i64 %x, 1
  ret i64 %a
}

; rdar://9166974
define i64 @t17(i16 %a, i64 %x) nounwind ssp {
entry:
; CHECK-LABEL: t17:
; CHECK: sxth [[REG:x[0-9]+]], w0
; CHECK: neg x0, [[REG]], lsl #32
; CHECK: ret
  %tmp16 = sext i16 %a to i64
  %tmp17 = mul i64 %tmp16, -4294967296
  ret i64 %tmp17
}

define i32 @t18(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t18:
; CHECK: sdiv w0, w0, w1
; CHECK: ret
  %sdiv = call i32 @llvm.aarch64.sdiv.i32(i32 %a, i32 %b)
  ret i32 %sdiv
}

define i64 @t19(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t19:
; CHECK: sdiv x0, x0, x1
; CHECK: ret
  %sdiv = call i64 @llvm.aarch64.sdiv.i64(i64 %a, i64 %b)
  ret i64 %sdiv
}

define i32 @t20(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t20:
; CHECK: udiv w0, w0, w1
; CHECK: ret
  %udiv = call i32 @llvm.aarch64.udiv.i32(i32 %a, i32 %b)
  ret i32 %udiv
}

define i64 @t21(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t21:
; CHECK: udiv x0, x0, x1
; CHECK: ret
  %udiv = call i64 @llvm.aarch64.udiv.i64(i64 %a, i64 %b)
  ret i64 %udiv
}

declare i32 @llvm.aarch64.sdiv.i32(i32, i32) nounwind readnone
declare i64 @llvm.aarch64.sdiv.i64(i64, i64) nounwind readnone
declare i32 @llvm.aarch64.udiv.i32(i32, i32) nounwind readnone
declare i64 @llvm.aarch64.udiv.i64(i64, i64) nounwind readnone

; 32-bit not.
define i32 @inv_32(i32 %x) nounwind ssp {
entry:
; CHECK: inv_32
; CHECK: mvn w0, w0
; CHECK: ret
  %inv = xor i32 %x, -1
  ret i32 %inv
}

; 64-bit not.
define i64 @inv_64(i64 %x) nounwind ssp {
entry:
; CHECK: inv_64
; CHECK: mvn x0, x0
; CHECK: ret
  %inv = xor i64 %x, -1
  ret i64 %inv
}

; Multiplying by a power of two plus or minus one is better done via shift
; and add/sub rather than the madd/msub instructions. The latter are 4+ cycles,
; and the former are two (total for the two instruction sequence for subtract).
define i32 @f0(i32 %a) nounwind readnone ssp {
; CHECK-LABEL: f0:
; CHECK-NEXT: add w0, w0, w0, lsl #3
; CHECK-NEXT: ret
  %res = mul i32 %a, 9
  ret i32 %res
}

define i64 @f1(i64 %a) nounwind readnone ssp {
; CHECK-LABEL: f1:
; CHECK-NEXT: lsl x8, x0, #4
; CHECK-NEXT: sub x0, x8, x0
; CHECK-NEXT: ret
  %res = mul i64 %a, 15
  ret i64 %res
}

define i32 @f2(i32 %a) nounwind readnone ssp {
; CHECK-LABEL: f2:
; CHECK-NEXT: lsl w8, w0, #3
; CHECK-NEXT: sub w0, w8, w0
; CHECK-NEXT: ret
  %res = mul nsw i32 %a, 7
  ret i32 %res
}

define i64 @f3(i64 %a) nounwind readnone ssp {
; CHECK-LABEL: f3:
; CHECK-NEXT: add x0, x0, x0, lsl #4
; CHECK-NEXT: ret
  %res = mul nsw i64 %a, 17
  ret i64 %res
}

define i32 @f4(i32 %a) nounwind readnone ssp {
; CHECK-LABEL: f4:
; CHECK-NEXT: add w0, w0, w0, lsl #1
; CHECK-NEXT: ret
  %res = mul i32 %a, 3
  ret i32 %res
}
