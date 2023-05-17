; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vnegs8(ptr %A) nounwind {
;CHECK-LABEL: vnegs8:
;CHECK: vneg.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = sub <8 x i8> zeroinitializer, %tmp1
	ret <8 x i8> %tmp2
}

define <4 x i16> @vnegs16(ptr %A) nounwind {
;CHECK-LABEL: vnegs16:
;CHECK: vneg.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = sub <4 x i16> zeroinitializer, %tmp1
	ret <4 x i16> %tmp2
}

define <2 x i32> @vnegs32(ptr %A) nounwind {
;CHECK-LABEL: vnegs32:
;CHECK: vneg.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = sub <2 x i32> zeroinitializer, %tmp1
	ret <2 x i32> %tmp2
}

define <2 x float> @vnegf32(ptr %A) nounwind {
;CHECK-LABEL: vnegf32:
;CHECK: vneg.f32
	%tmp1 = load <2 x float>, ptr %A
	%tmp2 = fsub <2 x float> < float -0.000000e+00, float -0.000000e+00 >, %tmp1
	ret <2 x float> %tmp2
}

define <16 x i8> @vnegQs8(ptr %A) nounwind {
;CHECK-LABEL: vnegQs8:
;CHECK: vneg.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = sub <16 x i8> zeroinitializer, %tmp1
	ret <16 x i8> %tmp2
}

define <8 x i16> @vnegQs16(ptr %A) nounwind {
;CHECK-LABEL: vnegQs16:
;CHECK: vneg.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = sub <8 x i16> zeroinitializer, %tmp1
	ret <8 x i16> %tmp2
}

define <4 x i32> @vnegQs32(ptr %A) nounwind {
;CHECK-LABEL: vnegQs32:
;CHECK: vneg.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = sub <4 x i32> zeroinitializer, %tmp1
	ret <4 x i32> %tmp2
}

define <4 x float> @vnegQf32(ptr %A) nounwind {
;CHECK-LABEL: vnegQf32:
;CHECK: vneg.f32
	%tmp1 = load <4 x float>, ptr %A
	%tmp2 = fsub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %tmp1
	ret <4 x float> %tmp2
}

define <8 x i8> @vqnegs8(ptr %A) nounwind {
;CHECK-LABEL: vqnegs8:
;CHECK: vqneg.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqneg.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqnegs16(ptr %A) nounwind {
;CHECK-LABEL: vqnegs16:
;CHECK: vqneg.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqneg.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqnegs32(ptr %A) nounwind {
;CHECK-LABEL: vqnegs32:
;CHECK: vqneg.s32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqneg.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <16 x i8> @vqnegQs8(ptr %A) nounwind {
;CHECK-LABEL: vqnegQs8:
;CHECK: vqneg.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqneg.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqnegQs16(ptr %A) nounwind {
;CHECK-LABEL: vqnegQs16:
;CHECK: vqneg.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqneg.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqnegQs32(ptr %A) nounwind {
;CHECK-LABEL: vqnegQs32:
;CHECK: vqneg.s32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqneg.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqneg.v8i8(<8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqneg.v4i16(<4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqneg.v2i32(<2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqneg.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqneg.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqneg.v4i32(<4 x i32>) nounwind readnone
