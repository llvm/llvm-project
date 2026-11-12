; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @rbit_8b(ptr %A) nounwind {
;CHECK-LABEL: rbit_8b:
;CHECK: rbit.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp3 = call <8 x i8> @llvm.bitreverse.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp3
}

define <16 x i8> @rbit_16b(ptr %A) nounwind {
;CHECK-LABEL: rbit_16b:
;CHECK: rbit.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp3 = call <16 x i8> @llvm.bitreverse.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp3
}

declare <8 x i8> @llvm.bitreverse.v8i8(<8 x i8>) nounwind readnone
declare <16 x i8> @llvm.bitreverse.v16i8(<16 x i8>) nounwind readnone

define <8 x i16> @sxtl8h(ptr %A) nounwind {
;CHECK-LABEL: sxtl8h:
;CHECK: sshll.8h
	%tmp1 = load <8 x i8>, ptr %A
  %tmp2 = sext <8 x i8> %tmp1 to <8 x i16>
  ret <8 x i16> %tmp2
}

define <8 x i16> @uxtl8h(ptr %A) nounwind {
;CHECK-LABEL: uxtl8h:
;CHECK: ushll.8h
	%tmp1 = load <8 x i8>, ptr %A
  %tmp2 = zext <8 x i8> %tmp1 to <8 x i16>
  ret <8 x i16> %tmp2
}

define <4 x i32> @sxtl4s(ptr %A) nounwind {
;CHECK-LABEL: sxtl4s:
;CHECK: sshll.4s
	%tmp1 = load <4 x i16>, ptr %A
  %tmp2 = sext <4 x i16> %tmp1 to <4 x i32>
  ret <4 x i32> %tmp2
}

define <4 x i32> @uxtl4s(ptr %A) nounwind {
;CHECK-LABEL: uxtl4s:
;CHECK: ushll.4s
	%tmp1 = load <4 x i16>, ptr %A
  %tmp2 = zext <4 x i16> %tmp1 to <4 x i32>
  ret <4 x i32> %tmp2
}

define <2 x i64> @sxtl2d(ptr %A) nounwind {
;CHECK-LABEL: sxtl2d:
;CHECK: sshll.2d
	%tmp1 = load <2 x i32>, ptr %A
  %tmp2 = sext <2 x i32> %tmp1 to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @uxtl2d(ptr %A) nounwind {
;CHECK-LABEL: uxtl2d:
;CHECK: ushll.2d
	%tmp1 = load <2 x i32>, ptr %A
  %tmp2 = zext <2 x i32> %tmp1 to <2 x i64>
  ret <2 x i64> %tmp2
}

; Check for incorrect use of vector bic.
; rdar://11553859
define void @test_vsliq(ptr nocapture %src, ptr nocapture %dest) nounwind noinline ssp {
entry:
; CHECK-LABEL: test_vsliq:
; CHECK-NOT: bic
; CHECK: movi.2d [[REG1:v[0-9]+]], #0x0000ff000000ff
; CHECK: and.16b v{{[0-9]+}}, v{{[0-9]+}}, [[REG1]]
  %0 = load <16 x i8>, ptr %src, align 16
  %and.i = and <16 x i8> %0, <i8 -1, i8 0, i8 0, i8 0, i8 -1, i8 0, i8 0, i8 0, i8 -1, i8 0, i8 0, i8 0, i8 -1, i8 0, i8 0, i8 0>
  %1 = bitcast <16 x i8> %and.i to <8 x i16>
  %vshl_n = shl <8 x i16> %1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %2 = or <8 x i16> %1, %vshl_n
  %3 = bitcast <8 x i16> %2 to <4 x i32>
  %vshl_n8 = shl <4 x i32> %3, <i32 16, i32 16, i32 16, i32 16>
  %4 = or <4 x i32> %3, %vshl_n8
  %5 = bitcast <4 x i32> %4 to <16 x i8>
  store <16 x i8> %5, ptr %dest, align 16
  ret void
}
