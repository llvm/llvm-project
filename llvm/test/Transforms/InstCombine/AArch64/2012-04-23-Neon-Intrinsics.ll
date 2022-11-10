; RUN: opt --mtriple=aarch64-unknown-linux -S -passes=instcombine < %s | FileCheck %s
; ARM64 neon intrinsic variants - <rdar://problem/12349617>

define <4 x i32> @mulByZeroARM64(<4 x i16> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %x, <4 x i16> zeroinitializer) nounwind
  ret <4 x i32> %a
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> zeroinitializer
}

define <4 x i32> @mulByOneARM64(<4 x i16> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %x, <4 x i16> <i16 1, i16 1, i16 1, i16 1>) nounwind
  ret <4 x i32> %a
; CHECK: entry:
; CHECK-NEXT: %a = sext <4 x i16> %x to <4 x i32>
; CHECK-NEXT: ret <4 x i32> %a
}

define <4 x i32> @constantMulARM64() nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> <i16 3, i16 3, i16 3, i16 3>, <4 x i16> <i16 2, i16 2, i16 2, i16 2>) nounwind
  ret <4 x i32> %a
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> <i32 6, i32 6, i32 6, i32 6>
}

define <4 x i32> @constantMulSARM64() nounwind readnone ssp {
entry:
  %b = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> <i16 1, i16 1, i16 1, i16 1>) nounwind
  ret <4 x i32> %b
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
}

define <4 x i32> @constantMulUARM64() nounwind readnone ssp {
entry:
  %b = tail call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> <i16 1, i16 1, i16 1, i16 1>) nounwind
  ret <4 x i32> %b
; CHECK: entry:
; CHECK-NEXT: ret <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
}

define <4 x i32> @complex1ARM64(<4 x i16> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> <i16 2, i16 2, i16 2, i16 2>, <4 x i16> %x) nounwind
  %b = add <4 x i32> zeroinitializer, %a
  ret <4 x i32> %b
; CHECK: entry:
; CHECK-NEXT: %a = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> <i16 2, i16 2, i16 2, i16 2>, <4 x i16> %x) [[NUW:#[0-9]+]]
; CHECK-NEXT: ret <4 x i32> %a
}

define <4 x i32> @complex2ARM64(<4 x i32> %x) nounwind readnone ssp {
entry:
  %a = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> <i16 3, i16 3, i16 3, i16 3>, <4 x i16> <i16 2, i16 2, i16 2, i16 2>) nounwind
  %b = add <4 x i32> %x, %a
  ret <4 x i32> %b
; CHECK: entry:
; CHECK-NEXT: %b = add <4 x i32> %x, <i32 6, i32 6, i32 6, i32 6>
; CHECK-NEXT: ret <4 x i32> %b
}

declare <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16>, <4 x i16>) nounwind readnone

; CHECK: attributes #0 = { nounwind ssp memory(none) }
; CHECK: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
; CHECK: attributes [[NUW]] = { nounwind }
