; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @subhn8b(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: subhn8b:
;CHECK: subhn.8b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.subhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @subhn4h(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: subhn4h:
;CHECK: subhn.4h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.subhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @subhn2s(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: subhn2s:
;CHECK: subhn.2s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.subhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i32> %tmp3
}

define <16 x i8> @subhn2_16b(<8 x i16> %a, <8 x i16> %b) nounwind  {
;CHECK-LABEL: subhn2_16b:
;CHECK: subhn.8b
;CHECK-NEXT: subhn2.16b
  %vsubhn2.i = tail call <8 x i8> @llvm.aarch64.neon.subhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %vsubhn_high2.i = tail call <8 x i8> @llvm.aarch64.neon.subhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %res = shufflevector <8 x i8> %vsubhn2.i, <8 x i8> %vsubhn_high2.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %res
}

define <8 x i16> @subhn2_8h(<4 x i32> %a, <4 x i32> %b) nounwind  {
;CHECK-LABEL: subhn2_8h:
;CHECK: subhn.4h
;CHECK-NEXT: subhn2.8h
  %vsubhn2.i = tail call <4 x i16> @llvm.aarch64.neon.subhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %vsubhn_high3.i = tail call <4 x i16> @llvm.aarch64.neon.subhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %res = shufflevector <4 x i16> %vsubhn2.i, <4 x i16> %vsubhn_high3.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %res
}

define <4 x i32> @subhn2_4s(<2 x i64> %a, <2 x i64> %b) nounwind  {
;CHECK-LABEL: subhn2_4s:
;CHECK: subhn.2s
;CHECK-NEXT: subhn2.4s
  %vsubhn2.i = tail call <2 x i32> @llvm.aarch64.neon.subhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %vsubhn_high3.i = tail call <2 x i32> @llvm.aarch64.neon.subhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %res = shufflevector <2 x i32> %vsubhn2.i, <2 x i32> %vsubhn_high3.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %res
}

declare <2 x i32> @llvm.aarch64.neon.subhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.subhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.subhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i8> @rsubhn8b(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: rsubhn8b:
;CHECK: rsubhn.8b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @rsubhn4h(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: rsubhn4h:
;CHECK: rsubhn.4h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @rsubhn2s(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: rsubhn2s:
;CHECK: rsubhn.2s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i32> %tmp3
}

define <16 x i8> @rsubhn2_16b(<8 x i16> %a, <8 x i16> %b) nounwind  {
;CHECK-LABEL: rsubhn2_16b:
;CHECK: rsubhn.8b
;CHECK-NEXT: rsubhn2.16b
  %vrsubhn2.i = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %vrsubhn_high2.i = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %res = shufflevector <8 x i8> %vrsubhn2.i, <8 x i8> %vrsubhn_high2.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %res
}

define <8 x i16> @rsubhn2_8h(<4 x i32> %a, <4 x i32> %b) nounwind  {
;CHECK-LABEL: rsubhn2_8h:
;CHECK: rsubhn.4h
;CHECK-NEXT: rsubhn2.8h
  %vrsubhn2.i = tail call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %vrsubhn_high3.i = tail call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %res = shufflevector <4 x i16> %vrsubhn2.i, <4 x i16> %vrsubhn_high3.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %res
}

define <4 x i32> @rsubhn2_4s(<2 x i64> %a, <2 x i64> %b) nounwind  {
;CHECK-LABEL: rsubhn2_4s:
;CHECK: rsubhn.2s
;CHECK-NEXT: rsubhn2.4s
  %vrsubhn2.i = tail call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %vrsubhn_high3.i = tail call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %res = shufflevector <2 x i32> %vrsubhn2.i, <2 x i32> %vrsubhn_high3.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %res
}

declare <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @ssubl8h(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: ssubl8h:
;CHECK: ssubl.8h
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
  %tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
  %tmp5 = sub <8 x i16> %tmp3, %tmp4
        ret <8 x i16> %tmp5
}

define <4 x i32> @ssubl4s(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: ssubl4s:
;CHECK: ssubl.4s
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
  %tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
  %tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
  %tmp5 = sub <4 x i32> %tmp3, %tmp4
        ret <4 x i32> %tmp5
}

define <2 x i64> @ssubl2d(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: ssubl2d:
;CHECK: ssubl.2d
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
  %tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
  %tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
  %tmp5 = sub <2 x i64> %tmp3, %tmp4
        ret <2 x i64> %tmp5
}

define <8 x i16> @ssubl2_8h(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: ssubl2_8h:
;CHECK: ssubl2.8h
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %high1 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %ext1 = sext <8 x i8> %high1 to <8 x i16>

        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %high2 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %ext2 = sext <8 x i8> %high2 to <8 x i16>

        %res = sub <8 x i16> %ext1, %ext2
        ret <8 x i16> %res
}

define <4 x i32> @ssubl2_4s(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: ssubl2_4s:
;CHECK: ssubl2.4s
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %high1 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %ext1 = sext <4 x i16> %high1 to <4 x i32>

        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %high2 = shufflevector <8 x i16> %tmp2, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %ext2 = sext <4 x i16> %high2 to <4 x i32>

        %res = sub <4 x i32> %ext1, %ext2
        ret <4 x i32> %res
}

define <2 x i64> @ssubl2_2d(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: ssubl2_2d:
;CHECK: ssubl2.2d
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %high1 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %ext1 = sext <2 x i32> %high1 to <2 x i64>

        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %high2 = shufflevector <4 x i32> %tmp2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %ext2 = sext <2 x i32> %high2 to <2 x i64>

        %res = sub <2 x i64> %ext1, %ext2
        ret <2 x i64> %res
}

define <8 x i16> @usubl8h(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: usubl8h:
;CHECK: usubl.8h
  %tmp1 = load <8 x i8>, <8 x i8>* %A
  %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
  %tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
  %tmp5 = sub <8 x i16> %tmp3, %tmp4
  ret <8 x i16> %tmp5
}

define <4 x i32> @usubl4s(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: usubl4s:
;CHECK: usubl.4s
  %tmp1 = load <4 x i16>, <4 x i16>* %A
  %tmp2 = load <4 x i16>, <4 x i16>* %B
  %tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
  %tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
  %tmp5 = sub <4 x i32> %tmp3, %tmp4
  ret <4 x i32> %tmp5
}

define <2 x i64> @usubl2d(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: usubl2d:
;CHECK: usubl.2d
  %tmp1 = load <2 x i32>, <2 x i32>* %A
  %tmp2 = load <2 x i32>, <2 x i32>* %B
  %tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
  %tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
  %tmp5 = sub <2 x i64> %tmp3, %tmp4
  ret <2 x i64> %tmp5
}

define <8 x i16> @usubl2_8h(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: usubl2_8h:
;CHECK: usubl2.8h
  %tmp1 = load <16 x i8>, <16 x i8>* %A
  %high1 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %ext1 = zext <8 x i8> %high1 to <8 x i16>

  %tmp2 = load <16 x i8>, <16 x i8>* %B
  %high2 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %ext2 = zext <8 x i8> %high2 to <8 x i16>

  %res = sub <8 x i16> %ext1, %ext2
  ret <8 x i16> %res
}

define <4 x i32> @usubl2_4s(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: usubl2_4s:
;CHECK: usubl2.4s
  %tmp1 = load <8 x i16>, <8 x i16>* %A
  %high1 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %ext1 = zext <4 x i16> %high1 to <4 x i32>

  %tmp2 = load <8 x i16>, <8 x i16>* %B
  %high2 = shufflevector <8 x i16> %tmp2, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %ext2 = zext <4 x i16> %high2 to <4 x i32>

  %res = sub <4 x i32> %ext1, %ext2
  ret <4 x i32> %res
}

define <2 x i64> @usubl2_2d(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: usubl2_2d:
;CHECK: usubl2.2d
  %tmp1 = load <4 x i32>, <4 x i32>* %A
  %high1 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %ext1 = zext <2 x i32> %high1 to <2 x i64>

  %tmp2 = load <4 x i32>, <4 x i32>* %B
  %high2 = shufflevector <4 x i32> %tmp2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %ext2 = zext <2 x i32> %high2 to <2 x i64>

  %res = sub <2 x i64> %ext1, %ext2
  ret <2 x i64> %res
}

define <8 x i16> @ssubw8h(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: ssubw8h:
;CHECK: ssubw.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = sext <8 x i8> %tmp2 to <8 x i16>
  %tmp4 = sub <8 x i16> %tmp1, %tmp3
        ret <8 x i16> %tmp4
}

define <4 x i32> @ssubw4s(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: ssubw4s:
;CHECK: ssubw.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
  %tmp3 = sext <4 x i16> %tmp2 to <4 x i32>
  %tmp4 = sub <4 x i32> %tmp1, %tmp3
        ret <4 x i32> %tmp4
}

define <2 x i64> @ssubw2d(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: ssubw2d:
;CHECK: ssubw.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
  %tmp3 = sext <2 x i32> %tmp2 to <2 x i64>
  %tmp4 = sub <2 x i64> %tmp1, %tmp3
        ret <2 x i64> %tmp4
}

define <8 x i16> @ssubw2_8h(<8 x i16>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: ssubw2_8h:
;CHECK: ssubw2.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A

        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %high2 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %ext2 = sext <8 x i8> %high2 to <8 x i16>

        %res = sub <8 x i16> %tmp1, %ext2
        ret <8 x i16> %res
}

define <4 x i32> @ssubw2_4s(<4 x i32>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: ssubw2_4s:
;CHECK: ssubw2.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A

        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %high2 = shufflevector <8 x i16> %tmp2, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %ext2 = sext <4 x i16> %high2 to <4 x i32>

        %res = sub <4 x i32> %tmp1, %ext2
        ret <4 x i32> %res
}

define <2 x i64> @ssubw2_2d(<2 x i64>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: ssubw2_2d:
;CHECK: ssubw2.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A

        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %high2 = shufflevector <4 x i32> %tmp2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %ext2 = sext <2 x i32> %high2 to <2 x i64>

        %res = sub <2 x i64> %tmp1, %ext2
        ret <2 x i64> %res
}

define <8 x i16> @usubw8h(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: usubw8h:
;CHECK: usubw.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = zext <8 x i8> %tmp2 to <8 x i16>
  %tmp4 = sub <8 x i16> %tmp1, %tmp3
        ret <8 x i16> %tmp4
}

define <4 x i32> @usubw4s(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: usubw4s:
;CHECK: usubw.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
  %tmp3 = zext <4 x i16> %tmp2 to <4 x i32>
  %tmp4 = sub <4 x i32> %tmp1, %tmp3
        ret <4 x i32> %tmp4
}

define <2 x i64> @usubw2d(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: usubw2d:
;CHECK: usubw.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
  %tmp3 = zext <2 x i32> %tmp2 to <2 x i64>
  %tmp4 = sub <2 x i64> %tmp1, %tmp3
        ret <2 x i64> %tmp4
}

define <8 x i16> @usubw2_8h(<8 x i16>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: usubw2_8h:
;CHECK: usubw2.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A

        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %high2 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %ext2 = zext <8 x i8> %high2 to <8 x i16>

        %res = sub <8 x i16> %tmp1, %ext2
        ret <8 x i16> %res
}

define <4 x i32> @usubw2_4s(<4 x i32>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: usubw2_4s:
;CHECK: usubw2.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A

        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %high2 = shufflevector <8 x i16> %tmp2, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %ext2 = zext <4 x i16> %high2 to <4 x i32>

        %res = sub <4 x i32> %tmp1, %ext2
        ret <4 x i32> %res
}

define <2 x i64> @usubw2_2d(<2 x i64>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: usubw2_2d:
;CHECK: usubw2.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A

        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %high2 = shufflevector <4 x i32> %tmp2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %ext2 = zext <2 x i32> %high2 to <2 x i64>

        %res = sub <2 x i64> %tmp1, %ext2
        ret <2 x i64> %res
}
