; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-eabi -mattr -neon -aarch64-neon-syntax=apple | FileCheck -check-prefix=CHECK-NONEON %s

define i32 @cnt32_advsimd(i32 %x) nounwind readnone {
  %cnt = tail call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %cnt
; CHECK: mov w[[IN64:[0-9]+]], w0
; CHECK: fmov	d0, x[[IN64]]
; CHECK: cnt.8b	v0, v0
; CHECK: uaddlv.8b	h0, v0
; CHECK: fmov w0, s0
; CHECK: ret
; CHECK-NONEON-LABEL: cnt32_advsimd
; CHECK-NONEON-NOT: 8b
; CHECK-NONEON: and w{{[0-9]+}}, w{{[0-9]+}}, #0x55555555
; CHECK-NONEON: and w{{[0-9]+}}, w{{[0-9]+}}, #0x33333333
; CHECK-NONEON: and w{{[0-9]+}}, w{{[0-9]+}}, #0xf0f0f0f
; CHECK-NONEON: mul
}

define i32 @cnt32_advsimd_2(<2 x i32> %x) {
  %1 = extractelement <2 x i32> %x, i64 0
  %2 = tail call i32 @llvm.ctpop.i32(i32 %1)
  ret i32 %2
; CHECK: fmov	w0, s0
; CHECK: fmov	d0, x0
; CHECK: cnt.8b	v0, v0
; CHECK: uaddlv.8b	h0, v0
; CHECK: fmov w0, s0
; CHECK: ret
; CHECK-NONEON-LABEL: cnt32_advsimd_2
; CHECK-NONEON-NOT: 8b
; CHECK-NONEON: and w{{[0-9]+}}, w{{[0-9]+}}, #0x55555555
; CHECK-NONEON: and w{{[0-9]+}}, w{{[0-9]+}}, #0x33333333
; CHECK-NONEON: and w{{[0-9]+}}, w{{[0-9]+}}, #0xf0f0f0f
; CHECK-NONEON: mul
}

define i64 @cnt64_advsimd(i64 %x) nounwind readnone {
  %cnt = tail call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %cnt
; CHECK: fmov	d0, x0
; CHECK: cnt.8b	v0, v0
; CHECK: uaddlv.8b	h0, v0
; CHECK: fmov	w0, s0
; CHECK: ret
; CHECK-NONEON-LABEL: cnt64_advsimd
; CHECK-NONEON-NOT: 8b
; CHECK-NONEON: and x{{[0-9]+}}, x{{[0-9]+}}, #0x5555555555555555
; CHECK-NONEON: and x{{[0-9]+}}, x{{[0-9]+}}, #0x3333333333333333
; CHECK-NONEON: and x{{[0-9]+}}, x{{[0-9]+}}, #0xf0f0f0f0f0f0f0f
; CHECK-NONEON: mul
}

; Do not use AdvSIMD when -mno-implicit-float is specified.
; rdar://9473858

define i32 @cnt32(i32 %x) nounwind readnone noimplicitfloat {
  %cnt = tail call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %cnt
; CHECK-LABEL: cnt32:
; CHECK-NOT: 16b
; CHECK: ret
}

define i64 @cnt64(i64 %x) nounwind readnone noimplicitfloat {
  %cnt = tail call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %cnt
; CHECK-LABEL: cnt64:
; CHECK-NOT: 16b
; CHECK: ret
}

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare i64 @llvm.ctpop.i64(i64) nounwind readnone
