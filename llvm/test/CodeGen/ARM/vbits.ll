; RUN: llc < %s -march=arm -mattr=+neon -mcpu=cortex-a8 | FileCheck %s

define <8 x i8> @v_andi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_andi8:
;CHECK: vand
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = and <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @v_andi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_andi16:
;CHECK: vand
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = and <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @v_andi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_andi32:
;CHECK: vand
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = and <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @v_andi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_andi64:
;CHECK: vand
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = and <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <16 x i8> @v_andQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_andQi8:
;CHECK: vand
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = and <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @v_andQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_andQi16:
;CHECK: vand
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = and <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @v_andQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_andQi32:
;CHECK: vand
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = and <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @v_andQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_andQi64:
;CHECK: vand
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = and <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <8 x i8> @v_bici8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_bici8:
;CHECK: vbic
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = xor <8 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = and <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @v_bici16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_bici16:
;CHECK: vbic
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = xor <4 x i16> %tmp2, < i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp4 = and <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @v_bici32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_bici32:
;CHECK: vbic
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = xor <2 x i32> %tmp2, < i32 -1, i32 -1 >
	%tmp4 = and <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @v_bici64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_bici64:
;CHECK: vbic
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = xor <1 x i64> %tmp2, < i64 -1 >
	%tmp4 = and <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @v_bicQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_bicQi8:
;CHECK: vbic
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = xor <16 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = and <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @v_bicQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_bicQi16:
;CHECK: vbic
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = xor <8 x i16> %tmp2, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp4 = and <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @v_bicQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_bicQi32:
;CHECK: vbic
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = xor <4 x i32> %tmp2, < i32 -1, i32 -1, i32 -1, i32 -1 >
	%tmp4 = and <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @v_bicQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_bicQi64:
;CHECK: vbic
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = xor <2 x i64> %tmp2, < i64 -1, i64 -1 >
	%tmp4 = and <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i8> @v_eori8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_eori8:
;CHECK: veor
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = xor <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @v_eori16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_eori16:
;CHECK: veor
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = xor <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @v_eori32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_eori32:
;CHECK: veor
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = xor <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @v_eori64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_eori64:
;CHECK: veor
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = xor <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <16 x i8> @v_eorQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_eorQi8:
;CHECK: veor
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = xor <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @v_eorQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_eorQi16:
;CHECK: veor
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = xor <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @v_eorQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_eorQi32:
;CHECK: veor
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = xor <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @v_eorQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_eorQi64:
;CHECK: veor
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = xor <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <8 x i8> @v_mvni8(<8 x i8>* %A) nounwind {
;CHECK: v_mvni8:
;CHECK: vmvn
	%tmp1 = load <8 x i8>* %A
	%tmp2 = xor <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @v_mvni16(<4 x i16>* %A) nounwind {
;CHECK: v_mvni16:
;CHECK: vmvn
	%tmp1 = load <4 x i16>* %A
	%tmp2 = xor <4 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @v_mvni32(<2 x i32>* %A) nounwind {
;CHECK: v_mvni32:
;CHECK: vmvn
	%tmp1 = load <2 x i32>* %A
	%tmp2 = xor <2 x i32> %tmp1, < i32 -1, i32 -1 >
	ret <2 x i32> %tmp2
}

define <1 x i64> @v_mvni64(<1 x i64>* %A) nounwind {
;CHECK: v_mvni64:
;CHECK: vmvn
	%tmp1 = load <1 x i64>* %A
	%tmp2 = xor <1 x i64> %tmp1, < i64 -1 >
	ret <1 x i64> %tmp2
}

define <16 x i8> @v_mvnQi8(<16 x i8>* %A) nounwind {
;CHECK: v_mvnQi8:
;CHECK: vmvn
	%tmp1 = load <16 x i8>* %A
	%tmp2 = xor <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @v_mvnQi16(<8 x i16>* %A) nounwind {
;CHECK: v_mvnQi16:
;CHECK: vmvn
	%tmp1 = load <8 x i16>* %A
	%tmp2 = xor <8 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @v_mvnQi32(<4 x i32>* %A) nounwind {
;CHECK: v_mvnQi32:
;CHECK: vmvn
	%tmp1 = load <4 x i32>* %A
	%tmp2 = xor <4 x i32> %tmp1, < i32 -1, i32 -1, i32 -1, i32 -1 >
	ret <4 x i32> %tmp2
}

define <2 x i64> @v_mvnQi64(<2 x i64>* %A) nounwind {
;CHECK: v_mvnQi64:
;CHECK: vmvn
	%tmp1 = load <2 x i64>* %A
	%tmp2 = xor <2 x i64> %tmp1, < i64 -1, i64 -1 >
	ret <2 x i64> %tmp2
}

define <8 x i8> @v_orri8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_orri8:
;CHECK: vorr
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = or <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @v_orri16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_orri16:
;CHECK: vorr
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = or <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @v_orri32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_orri32:
;CHECK: vorr
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = or <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @v_orri64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_orri64:
;CHECK: vorr
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = or <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <16 x i8> @v_orrQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_orrQi8:
;CHECK: vorr
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = or <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @v_orrQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_orrQi16:
;CHECK: vorr
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = or <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @v_orrQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_orrQi32:
;CHECK: vorr
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = or <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @v_orrQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_orrQi64:
;CHECK: vorr
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = or <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <8 x i8> @v_orni8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_orni8:
;CHECK: vorn
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = xor <8 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = or <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @v_orni16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_orni16:
;CHECK: vorn
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = xor <4 x i16> %tmp2, < i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp4 = or <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @v_orni32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_orni32:
;CHECK: vorn
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = xor <2 x i32> %tmp2, < i32 -1, i32 -1 >
	%tmp4 = or <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @v_orni64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_orni64:
;CHECK: vorn
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = xor <1 x i64> %tmp2, < i64 -1 >
	%tmp4 = or <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @v_ornQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_ornQi8:
;CHECK: vorn
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = xor <16 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = or <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @v_ornQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_ornQi16:
;CHECK: vorn
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = xor <8 x i16> %tmp2, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp4 = or <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @v_ornQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_ornQi32:
;CHECK: vorn
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = xor <4 x i32> %tmp2, < i32 -1, i32 -1, i32 -1, i32 -1 >
	%tmp4 = or <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @v_ornQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_ornQi64:
;CHECK: vorn
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = xor <2 x i64> %tmp2, < i64 -1, i64 -1 >
	%tmp4 = or <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i8> @vtsti8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vtsti8:
;CHECK: vtst.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = and <8 x i8> %tmp1, %tmp2
	%tmp4 = icmp ne <8 x i8> %tmp3, zeroinitializer
        %tmp5 = sext <8 x i1> %tmp4 to <8 x i8>
	ret <8 x i8> %tmp5
}

define <4 x i16> @vtsti16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vtsti16:
;CHECK: vtst.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = and <4 x i16> %tmp1, %tmp2
	%tmp4 = icmp ne <4 x i16> %tmp3, zeroinitializer
        %tmp5 = sext <4 x i1> %tmp4 to <4 x i16>
	ret <4 x i16> %tmp5
}

define <2 x i32> @vtsti32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vtsti32:
;CHECK: vtst.32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = and <2 x i32> %tmp1, %tmp2
	%tmp4 = icmp ne <2 x i32> %tmp3, zeroinitializer
        %tmp5 = sext <2 x i1> %tmp4 to <2 x i32>
	ret <2 x i32> %tmp5
}

define <16 x i8> @vtstQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vtstQi8:
;CHECK: vtst.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = and <16 x i8> %tmp1, %tmp2
	%tmp4 = icmp ne <16 x i8> %tmp3, zeroinitializer
        %tmp5 = sext <16 x i1> %tmp4 to <16 x i8>
	ret <16 x i8> %tmp5
}

define <8 x i16> @vtstQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vtstQi16:
;CHECK: vtst.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = and <8 x i16> %tmp1, %tmp2
	%tmp4 = icmp ne <8 x i16> %tmp3, zeroinitializer
        %tmp5 = sext <8 x i1> %tmp4 to <8 x i16>
	ret <8 x i16> %tmp5
}

define <4 x i32> @vtstQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vtstQi32:
;CHECK: vtst.32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = and <4 x i32> %tmp1, %tmp2
	%tmp4 = icmp ne <4 x i32> %tmp3, zeroinitializer
        %tmp5 = sext <4 x i1> %tmp4 to <4 x i32>
	ret <4 x i32> %tmp5
}

define <8 x i8> @v_orrimm(<8 x i8>* %A) nounwind {
; CHECK: v_orrimm:
; CHECK-NOT: vmov
; CHECK-NOT: vmvn
; CHECK: vorr
	%tmp1 = load <8 x i8>* %A
	%tmp3 = or <8 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1>
	ret <8 x i8> %tmp3
}

define <16 x i8> @v_orrimmQ(<16 x i8>* %A) nounwind {
; CHECK: v_orrimmQ
; CHECK-NOT: vmov
; CHECK-NOT: vmvn
; CHECK: vorr
	%tmp1 = load <16 x i8>* %A
	%tmp3 = or <16 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 1>
	ret <16 x i8> %tmp3
}

define <8 x i8> @v_bicimm(<8 x i8>* %A) nounwind {
; CHECK: v_bicimm:
; CHECK-NOT: vmov
; CHECK-NOT: vmvn
; CHECK: vbic
	%tmp1 = load <8 x i8>* %A
	%tmp3 = and <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0 >
	ret <8 x i8> %tmp3
}

define <16 x i8> @v_bicimmQ(<16 x i8>* %A) nounwind {
; CHECK: v_bicimmQ:
; CHECK-NOT: vmov
; CHECK-NOT: vmvn
; CHECK: vbic
	%tmp1 = load <16 x i8>* %A
	%tmp3 = and <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0 >
	ret <16 x i8> %tmp3
}
