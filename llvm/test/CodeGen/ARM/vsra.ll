; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vsras8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsras8:
;CHECK: vsra.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = ashr <8 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vsras16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsras16:
;CHECK: vsra.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = ashr <4 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vsras32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsras32:
;CHECK: vsra.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = ashr <2 x i32> %tmp2, < i32 32, i32 32 >
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vsras64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vsras64:
;CHECK: vsra.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = ashr <1 x i64> %tmp2, < i64 64 >
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @vsraQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vsraQs8:
;CHECK: vsra.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = ashr <16 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vsraQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vsraQs16:
;CHECK: vsra.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = ashr <8 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsraQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vsraQs32:
;CHECK: vsra.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = ashr <4 x i32> %tmp2, < i32 32, i32 32, i32 32, i32 32 >
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsraQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vsraQs64:
;CHECK: vsra.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = ashr <2 x i64> %tmp2, < i64 64, i64 64 >
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i8> @vsrau8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsrau8:
;CHECK: vsra.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = lshr <8 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vsrau16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsrau16:
;CHECK: vsra.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = lshr <4 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vsrau32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsrau32:
;CHECK: vsra.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = lshr <2 x i32> %tmp2, < i32 32, i32 32 >
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vsrau64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vsrau64:
;CHECK: vsra.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = lshr <1 x i64> %tmp2, < i64 64 >
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @vsraQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vsraQu8:
;CHECK: vsra.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = lshr <16 x i8> %tmp2, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vsraQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vsraQu16:
;CHECK: vsra.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = lshr <8 x i16> %tmp2, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsraQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vsraQu32:
;CHECK: vsra.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = lshr <4 x i32> %tmp2, < i32 32, i32 32, i32 32, i32 32 >
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsraQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vsraQu64:
;CHECK: vsra.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = lshr <2 x i64> %tmp2, < i64 64, i64 64 >
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i8> @vrsras8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vrsras8:
;CHECK: vrsra.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshifts.v8i8(<8 x i8> %tmp2, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vrsras16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vrsras16:
;CHECK: vrsra.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16> %tmp2, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vrsras32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vrsras32:
;CHECK: vrsra.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32> %tmp2, <2 x i32> < i32 -32, i32 -32 >)
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vrsras64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vrsras64:
;CHECK: vrsra.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64> %tmp2, <1 x i64> < i64 -64 >)
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <8 x i8> @vrsrau8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vrsrau8:
;CHECK: vrsra.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshiftu.v8i8(<8 x i8> %tmp2, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vrsrau16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vrsrau16:
;CHECK: vrsra.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16> %tmp2, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vrsrau32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vrsrau32:
;CHECK: vrsra.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32> %tmp2, <2 x i32> < i32 -32, i32 -32 >)
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @vrsrau64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vrsrau64:
;CHECK: vrsra.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64> %tmp2, <1 x i64> < i64 -64 >)
        %tmp4 = add <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @vrsraQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vrsraQs8:
;CHECK: vrsra.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8> %tmp2, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vrsraQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vrsraQs16:
;CHECK: vrsra.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16> %tmp2, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vrsraQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vrsraQs32:
;CHECK: vrsra.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32> %tmp2, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vrsraQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vrsraQs64:
;CHECK: vrsra.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64> %tmp2, <2 x i64> < i64 -64, i64 -64 >)
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <16 x i8> @vrsraQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vrsraQu8:
;CHECK: vrsra.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8> %tmp2, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vrsraQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vrsraQu16:
;CHECK: vrsra.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %tmp2, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vrsraQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vrsraQu32:
;CHECK: vrsra.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32> %tmp2, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vrsraQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vrsraQu64:
;CHECK: vrsra.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64> %tmp2, <2 x i64> < i64 -64, i64 -64 >)
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

declare <8 x i8>  @llvm.arm.neon.vrshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vrshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
