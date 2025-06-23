; RUN: llc -mtriple=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

define void @vshf_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.b [[R3]], [[R1]], [[R1]]
  store <16 x i8> %2, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @vshf_v16i8_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v16i8_1:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: splati.b [[R3:\$w[0-9]+]], [[R1]][1]
  store <16 x i8> %2, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @vshf_v16i8_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v16i8_2:

  %1 = load <16 x i8>, ptr %a
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 16>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.b [[R3]], [[R2]], [[R2]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @vshf_v16i8_3(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v16i8_3:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2, <16 x i32> <i32 17, i32 24, i32 25, i32 18, i32 19, i32 20, i32 28, i32 19, i32 1, i32 8, i32 9, i32 2, i32 3, i32 4, i32 12, i32 3>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.b [[R3]], [[R2]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @vshf_v16i8_4(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v16i8_4:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> %1, <16 x i32> <i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17, i32 1, i32 17>
  ; CHECK-DAG: splati.b [[R3:\$w[0-9]+]], [[R1]][1]
  store <16 x i8> %2, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @vshf_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.h [[R3]], [[R1]], [[R1]]
  store <8 x i16> %2, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @vshf_v8i16_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v8i16_1:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: splati.h [[R3:\$w[0-9]+]], [[R1]][1]
  store <8 x i16> %2, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @vshf_v8i16_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v8i16_2:

  %1 = load <8 x i16>, ptr %a
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 8>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.h [[R3]], [[R2]], [[R2]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @vshf_v8i16_3(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v8i16_3:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 8, i32 9, i32 2, i32 3, i32 4, i32 12, i32 3>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.h [[R3]], [[R2]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @vshf_v8i16_4(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v8i16_4:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> %1, <8 x i32> <i32 1, i32 9, i32 1, i32 9, i32 1, i32 9, i32 1, i32 9>
  ; CHECK-DAG: splati.h [[R3:\$w[0-9]+]], [[R1]][1]
  store <8 x i16> %2, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

; Note: v4i32 only has one 4-element set so it's impossible to get a vshf.w
; instruction when using a single vector.

define void @vshf_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R1]], 27
  store <4 x i32> %2, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @vshf_v4i32_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v4i32_1:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: splati.w [[R3:\$w[0-9]+]], [[R1]][1]
  store <4 x i32> %2, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @vshf_v4i32_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v4i32_2:

  %1 = load <4 x i32>, ptr %a
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 4, i32 5, i32 6, i32 4>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R2]], 36
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @vshf_v4i32_3(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v4i32_3:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 5, i32 6, i32 4>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.w [[R3]], [[R2]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @vshf_v4i32_4(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v4i32_4:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> %1, <4 x i32> <i32 1, i32 5, i32 5, i32 1>
  ; The two operand vectors are the same so element 1 and 5 are equivalent.
  ; CHECK-DAG: splati.w [[R3:\$w[0-9]+]], [[R1]][1]
  store <4 x i32> %2, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @vshf_v2i64_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.d [[R3]], [[R1]], [[R1]]
  store <2 x i64> %2, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @vshf_v2i64_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v2i64_1:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %2, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @vshf_v2i64_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v2i64_2:

  %1 = load <2 x i64>, ptr %a
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 3, i32 2>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; CHECK-DAG: vshf.d [[R3]], [[R2]], [[R2]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @vshf_v2i64_3(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v2i64_3:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 2>
  ; CHECK-DAG: addiu [[PTR_A:\$[0-9]+]], {{.*}}, %lo($
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0([[PTR_A]])
  ; The concatenation step of vshf is bitwise not vectorwise so we must reverse
  ; the operands to get the right answer.
  ; CHECK-DAG: vshf.d [[R3]], [[R2]], [[R1]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @vshf_v2i64_4(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: vshf_v2i64_4:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> %1, <2 x i32> <i32 1, i32 3>
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %2, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @shf_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: shf_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> <i32 1, i32 3, i32 2, i32 0, i32 5, i32 7, i32 6, i32 4, i32 9, i32 11, i32 10, i32 8, i32 13, i32 15, i32 14, i32 12>
  ; CHECK-DAG: shf.b [[R3:\$w[0-9]+]], [[R1]], 45
  store <16 x i8> %2, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @shf_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: shf_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ; CHECK-DAG: shf.h [[R3:\$w[0-9]+]], [[R1]], 27
  store <8 x i16> %2, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @shf_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: shf_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ; CHECK-DAG: shf.w [[R3:\$w[0-9]+]], [[R1]], 27
  store <4 x i32> %2, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

; shf.d does not exist

define void @ilvev_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  ; CHECK-DAG: ilvev.b [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvev_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ; CHECK-DAG: ilvev.h [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvev_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ; CHECK-DAG: ilvev.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvev_v2i64_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  ; CHECK-DAG: ilvev.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

; Interleaving one operand with itself.
define void @ilvev_v16i8_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v16i8_1:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14>
  ; CHECK-DAG: ilvev.b [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvev_v8i16_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v8i16_1:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ; CHECK-DAG: ilvev.h [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvev_v4i32_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v4i32_1:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ; CHECK-DAG: ilvev.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvev_v2i64_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v2i64_1:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 0>
  ; ilvev.d with two identical operands is equivalent to splati.d
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][0]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvev_v16i8_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v16i8_2:

  %1 = load <16 x i8>, ptr %a
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 16, i32 16, i32 18, i32 18, i32 20, i32 20, i32 22, i32 22, i32 24, i32 24, i32 26, i32 26, i32 28, i32 28, i32 30, i32 30>
  ; CHECK-DAG: ilvev.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvev_v8i16_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v8i16_2:

  %1 = load <8 x i16>, ptr %a
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14>
  ; CHECK-DAG: ilvev.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvev_v4i32_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v4i32_2:

  %1 = load <4 x i32>, ptr %a
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 4, i32 4, i32 6, i32 6>
  ; CHECK-DAG: ilvev.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvev_v2i64_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvev_v2i64_2:

  %1 = load <2 x i64>, ptr %a
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 2, i32 2>
  ; ilvev.d with two identical operands is equivalent to splati.d
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R2]][0]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvod_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  ; CHECK-DAG: ilvod.b [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvod_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ; CHECK-DAG: ilvod.h [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvod_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ; CHECK-DAG: ilvod.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvod_v2i64_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 3>
  ; CHECK-DAG: ilvod.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvod_v16i8_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v16i8_1:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  ; CHECK-DAG: ilvod.b [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvod_v8i16_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v8i16_1:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ; CHECK-DAG: ilvod.h [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvod_v4i32_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v4i32_1:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  ; CHECK-DAG: ilvod.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvod_v2i64_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v2i64_1:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 1>
  ; ilvod.d with two identical operands is equivalent to splati.d
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvod_v16i8_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v16i8_2:

  %1 = load <16 x i8>, ptr %a
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 17, i32 17, i32 19, i32 19, i32 21, i32 21, i32 23, i32 23, i32 25, i32 25, i32 27, i32 27, i32 29, i32 29, i32 31, i32 31>
  ; CHECK-DAG: ilvod.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvod_v8i16_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v8i16_2:

  %1 = load <8 x i16>, ptr %a
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  ; CHECK-DAG: ilvod.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvod_v4i32_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v4i32_2:

  %1 = load <4 x i32>, ptr %a
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 5, i32 5, i32 7, i32 7>
  ; CHECK-DAG: ilvod.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvod_v2i64_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvod_v2i64_2:

  %1 = load <2 x i64>, ptr %a
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 3, i32 3>
  ; ilvod.d with two identical operands is equivalent to splati.d
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R2]][1]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvr_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ; CHECK-DAG: ilvr.b [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvr_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ; CHECK-DAG: ilvr.h [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvr_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ; CHECK-DAG: ilvr.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvr_v2i64_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  ; ilvr.d and ilvev.d are equivalent for v2i64
  ; CHECK-DAG: ilvev.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvr_v16i8_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v16i8_1:

  %1 = load <16 x i8>, ptr %a
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 16, i32 16, i32 17, i32 17, i32 18, i32 18, i32 19, i32 19, i32 20, i32 20, i32 21, i32 21, i32 22, i32 22, i32 23, i32 23>
  ; CHECK-DAG: ilvr.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvr_v8i16_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v8i16_1:

  %1 = load <8 x i16>, ptr %a
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11>
  ; CHECK-DAG: ilvr.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvr_v4i32_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v4i32_1:

  %1 = load <4 x i32>, ptr %a
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 4, i32 4, i32 5, i32 5>
  ; CHECK-DAG: ilvr.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvr_v2i64_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v2i64_1:

  %1 = load <2 x i64>, ptr %a
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 2, i32 2>
  ; ilvr.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R2]][0]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvr_v16i8_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v16i8_2:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  ; CHECK-DAG: ilvr.b [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvr_v8i16_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v8i16_2:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ; CHECK-DAG: ilvr.h [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvr_v4i32_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v4i32_2:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  ; CHECK-DAG: ilvr.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvr_v2i64_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvr_v2i64_2:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 0>
  ; ilvr.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][0]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvl_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ; CHECK-DAG: ilvl.b [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvl_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ; CHECK-DAG: ilvl.h [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvl_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ; CHECK-DAG: ilvl.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvl_v2i64_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 3>
  ; ilvl.d and ilvod.d are equivalent for v2i64
  ; CHECK-DAG: ilvod.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvl_v16i8_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v16i8_1:

  %1 = load <16 x i8>, ptr %a
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 24, i32 24, i32 25, i32 25, i32 26, i32 26, i32 27, i32 27, i32 28, i32 28, i32 29, i32 29, i32 30, i32 30, i32 31, i32 31>
  ; CHECK-DAG: ilvl.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvl_v8i16_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v8i16_1:

  %1 = load <8 x i16>, ptr %a
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
  ; CHECK-DAG: ilvl.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvl_v4i32_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v4i32_1:

  %1 = load <4 x i32>, ptr %a
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 6, i32 6, i32 7, i32 7>
  ; CHECK-DAG: ilvl.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvl_v2i64_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v2i64_1:

  %1 = load <2 x i64>, ptr %a
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 3, i32 3>
  ; ilvl.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R2]][1]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @ilvl_v16i8_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v16i8_2:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11, i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
  ; CHECK-DAG: ilvl.b [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @ilvl_v8i16_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v8i16_2:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  ; CHECK-DAG: ilvl.h [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @ilvl_v4i32_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v4i32_2:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 2, i32 2, i32 3, i32 3>
  ; CHECK-DAG: ilvl.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @ilvl_v2i64_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: ilvl_v2i64_2:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 1>
  ; ilvl.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @pckev_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  ; CHECK-DAG: pckev.b [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @pckev_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ; CHECK-DAG: pckev.h [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @pckev_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ; CHECK-DAG: pckev.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @pckev_v2i64_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  ; pckev.d and ilvev.d are equivalent for v2i64
  ; CHECK-DAG: ilvev.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @pckev_v16i8_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v16i8_1:

  %1 = load <16 x i8>, ptr %a
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  ; CHECK-DAG: pckev.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @pckev_v8i16_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v8i16_1:

  %1 = load <8 x i16>, ptr %a
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 8, i32 10, i32 12, i32 14, i32 8, i32 10, i32 12, i32 14>
  ; CHECK-DAG: pckev.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @pckev_v4i32_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v4i32_1:

  %1 = load <4 x i32>, ptr %a
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 4, i32 6, i32 4, i32 6>
  ; CHECK-DAG: pckev.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @pckev_v2i64_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v2i64_1:

  %1 = load <2 x i64>, ptr %a
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 2, i32 2>
  ; pckev.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R2]][0]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @pckev_v16i8_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v16i8_2:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  ; CHECK-DAG: pckev.b [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @pckev_v8i16_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v8i16_2:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 0, i32 2, i32 4, i32 6>
  ; CHECK-DAG: pckev.h [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @pckev_v4i32_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v4i32_2:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 2, i32 0, i32 2>
  ; CHECK-DAG: pckev.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @pckev_v2i64_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckev_v2i64_2:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 0>
  ; pckev.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][0]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @pckod_v16i8_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ; CHECK-DAG: pckod.b [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @pckod_v8i16_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ; CHECK-DAG: pckod.h [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @pckod_v4i32_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ; CHECK-DAG: pckod.w [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @pckod_v2i64_0(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 3>
  ; pckod.d and ilvod.d are equivalent for v2i64
  ; CHECK-DAG: ilvod.d [[R3:\$w[0-9]+]], [[R2]], [[R1]]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @pckod_v16i8_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v16i8_1:

  %1 = load <16 x i8>, ptr %a
  %2 = load <16 x i8>, ptr %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  ; CHECK-DAG: pckod.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @pckod_v8i16_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v8i16_1:

  %1 = load <8 x i16>, ptr %a
  %2 = load <8 x i16>, ptr %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 9, i32 11, i32 13, i32 15, i32 9, i32 11, i32 13, i32 15>
  ; CHECK-DAG: pckod.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @pckod_v4i32_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v4i32_1:

  %1 = load <4 x i32>, ptr %a
  %2 = load <4 x i32>, ptr %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 5, i32 7, i32 5, i32 7>
  ; CHECK-DAG: pckod.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @pckod_v2i64_1(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v2i64_1:

  %1 = load <2 x i64>, ptr %a
  %2 = load <2 x i64>, ptr %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 3, i32 3>
  ; pckod.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R2]][1]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @pckod_v16i8_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v16i8_2:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, ptr %b
  %3 = shufflevector <16 x i8> %1, <16 x i8> %2,
                     <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ; CHECK-DAG: pckod.b [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <16 x i8> %3, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @pckod_v8i16_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v8i16_2:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, ptr %b
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 1, i32 3, i32 5, i32 7>
  ; CHECK-DAG: pckod.h [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <8 x i16> %3, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @pckod_v4i32_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v4i32_2:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, ptr %b
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 1, i32 3, i32 1, i32 3>
  ; CHECK-DAG: pckod.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <4 x i32> %3, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @pckod_v2i64_2(ptr %c, ptr %a, ptr %b) nounwind {
  ; CHECK-LABEL: pckod_v2i64_2:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, ptr %b
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 1, i32 1>
  ; pckod.d and splati.d are equivalent for v2i64
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %3, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}

define void @splati_v16i8_0(ptr %c, ptr %a) nounwind {
  ; CHECK-LABEL: splati_v16i8_0:

  %1 = load <16 x i8>, ptr %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef,
                     <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ; CHECK-DAG: splati.b [[R3:\$w[0-9]+]], [[R1]][4]
  store <16 x i8> %2, ptr %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
}

define void @splati_v8i16_0(ptr %c, ptr %a) nounwind {
  ; CHECK-LABEL: splati_v8i16_0:

  %1 = load <8 x i16>, ptr %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ; CHECK-DAG: splati.h [[R3:\$w[0-9]+]], [[R1]][4]
  store <8 x i16> %2, ptr %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
}

define void @splati_v4i32_0(ptr %c, ptr %a) nounwind {
  ; CHECK-LABEL: splati_v4i32_0:

  %1 = load <4 x i32>, ptr %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  ; CHECK-DAG: splati.w [[R3:\$w[0-9]+]], [[R1]][3]
  store <4 x i32> %2, ptr %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
}

define void @splati_v2i64_0(ptr %c, ptr %a) nounwind {
  ; CHECK-LABEL: splati_v2i64_0:

  %1 = load <2 x i64>, ptr %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ; CHECK-DAG: splati.d [[R3:\$w[0-9]+]], [[R1]][1]
  store <2 x i64> %2, ptr %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
}
