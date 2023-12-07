; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vmins8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmins8:
;CHECK: vmin.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmins.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vmins16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmins16:
;CHECK: vmin.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vmins.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vmins32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmins32:
;CHECK: vmin.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vmins.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @vminu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminu8:
;CHECK: vmin.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vminu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vminu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminu16:
;CHECK: vmin.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vminu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vminu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminu32:
;CHECK: vmin.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vminu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <2 x float> @vminf32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminf32:
;CHECK: vmin.f32
	%tmp1 = load <2 x float>, ptr %A
	%tmp2 = load <2 x float>, ptr %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vmins.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <16 x i8> @vminQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminQs8:
;CHECK: vmin.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmins.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vminQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminQs16:
;CHECK: vmin.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmins.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vminQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminQs32:
;CHECK: vmin.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <16 x i8> @vminQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminQu8:
;CHECK: vmin.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vminu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vminQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminQu16:
;CHECK: vmin.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vminu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vminQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminQu32:
;CHECK: vmin.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <4 x float> @vminQf32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vminQf32:
;CHECK: vmin.f32
	%tmp1 = load <4 x float>, ptr %A
	%tmp2 = load <4 x float>, ptr %B
	%tmp3 = call <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmins.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vmins.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vmins.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vminu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vminu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vminu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vmins.v2f32(<2 x float>, <2 x float>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vmins.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vmins.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vminu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vminu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <8 x i8> @vmaxs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxs8:
;CHECK: vmax.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmaxs.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vmaxs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxs16:
;CHECK: vmax.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vmaxs.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vmaxs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxs32:
;CHECK: vmax.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vmaxs.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i8> @vmaxu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxu8:
;CHECK: vmax.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmaxu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vmaxu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxu16:
;CHECK: vmax.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vmaxu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vmaxu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxu32:
;CHECK: vmax.u32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vmaxu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <2 x float> @vmaxf32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxf32:
;CHECK: vmax.f32
	%tmp1 = load <2 x float>, ptr %A
	%tmp2 = load <2 x float>, ptr %B
	%tmp3 = call <2 x float> @llvm.arm.neon.vmaxs.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

define <16 x i8> @vmaxQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxQs8:
;CHECK: vmax.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmaxs.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vmaxQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxQs16:
;CHECK: vmax.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmaxs.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmaxQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxQs32:
;CHECK: vmax.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <16 x i8> @vmaxQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxQu8:
;CHECK: vmax.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmaxu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vmaxQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxQu16:
;CHECK: vmax.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmaxQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxQu32:
;CHECK: vmax.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <4 x float> @vmaxQf32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vmaxQf32:
;CHECK: vmax.f32
	%tmp1 = load <4 x float>, ptr %A
	%tmp2 = load <4 x float>, ptr %B
	%tmp3 = call <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmaxs.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vmaxs.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vmaxs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vmaxu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vmaxu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vmaxu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vmaxs.v2f32(<2 x float>, <2 x float>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vmaxs.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vmaxs.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vmaxu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>) nounwind readnone

declare float @llvm.maxnum.f32(float %a, float %b)
declare float @llvm.minnum.f32(float %a, float %b)

define float @maxnum(float %a, float %b) {
;CHECK-LABEL: maxnum:
;CHECK: vcmp.f32
;CHECK-NEXT: vmrs
;CHECK-NEXT: vmovgt.f32
  %r = call nnan float @llvm.maxnum.f32(float %a, float %b)
  ret float %r
}

define float @minnum(float %a, float %b) {
;CHECK-LABEL: minnum:
;CHECK: vcmp.f32
;CHECK-NEXT: vmrs
;CHECK-NEXT: vmovlt.f32
  %r = call nnan float @llvm.minnum.f32(float %a, float %b)
  ret float %r
}
