; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vshls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vshls8:
;CHECK: vshl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shl <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vshls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vshls16:
;CHECK: vshl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = shl <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vshls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vshls32:
;CHECK: vshl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = shl <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vshls64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vshls64:
;CHECK: vshl.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = shl <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vshli8(<8 x i8>* %A) nounwind {
;CHECK: vshli8:
;CHECK: vshl.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shl <8 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vshli16(<4 x i16>* %A) nounwind {
;CHECK: vshli16:
;CHECK: vshl.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shl <4 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vshli32(<2 x i32>* %A) nounwind {
;CHECK: vshli32:
;CHECK: vshl.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = shl <2 x i32> %tmp1, < i32 31, i32 31 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vshli64(<1 x i64>* %A) nounwind {
;CHECK: vshli64:
;CHECK: vshl.i64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = shl <1 x i64> %tmp1, < i64 63 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vshlQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vshlQs8:
;CHECK: vshl.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = shl <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vshlQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vshlQs16:
;CHECK: vshl.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shl <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vshlQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vshlQs32:
;CHECK: vshl.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = shl <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vshlQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vshlQs64:
;CHECK: vshl.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = shl <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vshlQi8(<16 x i8>* %A) nounwind {
;CHECK: vshlQi8:
;CHECK: vshl.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shl <16 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vshlQi16(<8 x i16>* %A) nounwind {
;CHECK: vshlQi16:
;CHECK: vshl.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shl <8 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vshlQi32(<4 x i32>* %A) nounwind {
;CHECK: vshlQi32:
;CHECK: vshl.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = shl <4 x i32> %tmp1, < i32 31, i32 31, i32 31, i32 31 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vshlQi64(<2 x i64>* %A) nounwind {
;CHECK: vshlQi64:
;CHECK: vshl.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = shl <2 x i64> %tmp1, < i64 63, i64 63 >
	ret <2 x i64> %tmp2
}

define <8 x i8> @vlshru8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vlshru8:
;CHECK: vneg.s8
;CHECK: vshl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = lshr <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vlshru16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vlshru16:
;CHECK: vneg.s16
;CHECK: vshl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = lshr <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vlshru32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vlshru32:
;CHECK: vneg.s32
;CHECK: vshl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = lshr <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vlshru64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vlshru64:
;CHECK: vsub.i64
;CHECK: vshl.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = lshr <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vlshri8(<8 x i8>* %A) nounwind {
;CHECK: vlshri8:
;CHECK: vshr.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = lshr <8 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vlshri16(<4 x i16>* %A) nounwind {
;CHECK: vlshri16:
;CHECK: vshr.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = lshr <4 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vlshri32(<2 x i32>* %A) nounwind {
;CHECK: vlshri32:
;CHECK: vshr.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = lshr <2 x i32> %tmp1, < i32 32, i32 32 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vlshri64(<1 x i64>* %A) nounwind {
;CHECK: vlshri64:
;CHECK: vshr.u64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = lshr <1 x i64> %tmp1, < i64 64 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vlshrQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vlshrQu8:
;CHECK: vneg.s8
;CHECK: vshl.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = lshr <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vlshrQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vlshrQu16:
;CHECK: vneg.s16
;CHECK: vshl.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = lshr <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vlshrQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vlshrQu32:
;CHECK: vneg.s32
;CHECK: vshl.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = lshr <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vlshrQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vlshrQu64:
;CHECK: vsub.i64
;CHECK: vshl.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = lshr <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vlshrQi8(<16 x i8>* %A) nounwind {
;CHECK: vlshrQi8:
;CHECK: vshr.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = lshr <16 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vlshrQi16(<8 x i16>* %A) nounwind {
;CHECK: vlshrQi16:
;CHECK: vshr.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = lshr <8 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vlshrQi32(<4 x i32>* %A) nounwind {
;CHECK: vlshrQi32:
;CHECK: vshr.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = lshr <4 x i32> %tmp1, < i32 32, i32 32, i32 32, i32 32 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vlshrQi64(<2 x i64>* %A) nounwind {
;CHECK: vlshrQi64:
;CHECK: vshr.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = lshr <2 x i64> %tmp1, < i64 64, i64 64 >
	ret <2 x i64> %tmp2
}

; Example that requires splitting and expanding a vector shift.
define <2 x i64> @update(<2 x i64> %val) nounwind readnone {
entry:
	%shr = lshr <2 x i64> %val, < i64 2, i64 2 >		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %shr
}

define <8 x i8> @vashrs8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vashrs8:
;CHECK: vneg.s8
;CHECK: vshl.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = ashr <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vashrs16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vashrs16:
;CHECK: vneg.s16
;CHECK: vshl.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = ashr <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vashrs32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vashrs32:
;CHECK: vneg.s32
;CHECK: vshl.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = ashr <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vashrs64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vashrs64:
;CHECK: vsub.i64
;CHECK: vshl.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = ashr <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <8 x i8> @vashri8(<8 x i8>* %A) nounwind {
;CHECK: vashri8:
;CHECK: vshr.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = ashr <8 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vashri16(<4 x i16>* %A) nounwind {
;CHECK: vashri16:
;CHECK: vshr.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = ashr <4 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vashri32(<2 x i32>* %A) nounwind {
;CHECK: vashri32:
;CHECK: vshr.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = ashr <2 x i32> %tmp1, < i32 32, i32 32 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @vashri64(<1 x i64>* %A) nounwind {
;CHECK: vashri64:
;CHECK: vshr.s64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = ashr <1 x i64> %tmp1, < i64 64 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @vashrQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vashrQs8:
;CHECK: vneg.s8
;CHECK: vshl.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = ashr <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vashrQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vashrQs16:
;CHECK: vneg.s16
;CHECK: vshl.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = ashr <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vashrQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vashrQs32:
;CHECK: vneg.s32
;CHECK: vshl.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = ashr <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vashrQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vashrQs64:
;CHECK: vsub.i64
;CHECK: vshl.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = ashr <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <16 x i8> @vashrQi8(<16 x i8>* %A) nounwind {
;CHECK: vashrQi8:
;CHECK: vshr.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = ashr <16 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vashrQi16(<8 x i16>* %A) nounwind {
;CHECK: vashrQi16:
;CHECK: vshr.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = ashr <8 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vashrQi32(<4 x i32>* %A) nounwind {
;CHECK: vashrQi32:
;CHECK: vshr.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = ashr <4 x i32> %tmp1, < i32 32, i32 32, i32 32, i32 32 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @vashrQi64(<2 x i64>* %A) nounwind {
;CHECK: vashrQi64:
;CHECK: vshr.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = ashr <2 x i64> %tmp1, < i64 64, i64 64 >
	ret <2 x i64> %tmp2
}
