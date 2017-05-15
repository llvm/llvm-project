; RUN: opt -S -instcombine < %s | FileCheck %s

define <4 x i32> @mulByZero(<4 x i16> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %x, <4 x i16> zeroinitializer) nounwind
  ret <4 x i32> %a
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> zeroinitializer
}

define <4 x i32> @mulByOne(<4 x i16> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %x, <4 x i16> <i16 1, i16 1, i16 1, i16 1>) nounwind
  ret <4 x i32> %a
; CHECK: entry:
; CHECK-NEXT: %a = sext <4 x i16> %x to <4 x i32>
; CHECK-NEXT: ret <4 x i32> %a
}

define <4 x i32> @constantMul() nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> <i16 3, i16 3, i16 3, i16 3>, <4 x i16> <i16 2, i16 2, i16 2, i16 2>) nounwind
  ret <4 x i32> %a
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> <i32 6, i32 6, i32 6, i32 6>
}

define <4 x i32> @constantMulS() nounwind readnone ssp {
entry:
  %b = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> <i16 1, i16 1, i16 1, i16 1>) nounwind
  ret <4 x i32> %b
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
}

define <4 x i32> @constantMulU() nounwind readnone ssp {
entry:
  %b = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> <i16 1, i16 1, i16 1, i16 1>) nounwind
  ret <4 x i32> %b
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
}

define <4 x i32> @complex1(<4 x i16> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> <i16 2, i16 2, i16 2, i16 2>, <4 x i16> %x) nounwind
  %b = add <4 x i32> zeroinitializer, %a
  ret <4 x i32> %b
; CHECK: entry:
; CHECK-NEXT: %a = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> <i16 2, i16 2, i16 2, i16 2>, <4 x i16> %x) [[NUW:#[0-9]+]]
; CHECK-NEXT: ret <4 x i32> %a
}

define <4 x i32> @complex2(<4 x i32> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> <i16 3, i16 3, i16 3, i16 3>, <4 x i16> <i16 2, i16 2, i16 2, i16 2>) nounwind
  %b = add <4 x i32> %x, %a
  ret <4 x i32> %b  
; CHECK: entry:
; CHECK-NEXT: %b = add <4 x i32> %x, <i32 6, i32 6, i32 6, i32 6>
; CHECK-NEXT: ret <4 x i32> %b
}

declare <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
