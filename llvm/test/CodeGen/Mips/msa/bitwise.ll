; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

define void @and_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: and_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = and <16 x i8> %1, %2
  ; CHECK-DAG: and.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size and_v16i8
}

define void @and_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: and_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = and <8 x i16> %1, %2
  ; CHECK-DAG: and.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size and_v8i16
}

define void @and_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: and_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = and <4 x i32> %1, %2
  ; CHECK-DAG: and.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size and_v4i32
}

define void @and_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: and_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = and <2 x i64> %1, %2
  ; CHECK-DAG: and.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size and_v2i64
}

define void @and_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: and_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = and <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: andi.b [[R4:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size and_v16i8_i
}

define void @and_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: and_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = and <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: and.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size and_v8i16_i
}

define void @and_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: and_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = and <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: and.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size and_v4i32_i
}

define void @and_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: and_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = and <2 x i64> %1, <i64 1, i64 1>
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: and.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size and_v2i64_i
}

define void @or_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: or_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = or <16 x i8> %1, %2
  ; CHECK-DAG: or.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size or_v16i8
}

define void @or_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: or_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = or <8 x i16> %1, %2
  ; CHECK-DAG: or.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size or_v8i16
}

define void @or_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: or_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = or <4 x i32> %1, %2
  ; CHECK-DAG: or.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size or_v4i32
}

define void @or_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: or_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = or <2 x i64> %1, %2
  ; CHECK-DAG: or.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size or_v2i64
}

define void @or_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: or_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = or <16 x i8> %1, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ; CHECK-DAG: ori.b [[R4:\$w[0-9]+]], [[R1]], 3
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size or_v16i8_i
}

define void @or_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: or_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = or <8 x i16> %1, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 3
  ; CHECK-DAG: or.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size or_v8i16_i
}

define void @or_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: or_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = or <4 x i32> %1, <i32 3, i32 3, i32 3, i32 3>
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 3
  ; CHECK-DAG: or.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size or_v4i32_i
}

define void @or_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: or_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = or <2 x i64> %1, <i64 3, i64 3>
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 3
  ; CHECK-DAG: or.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size or_v2i64_i
}

define void @nor_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: nor_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = or <16 x i8> %1, %2
  %4 = xor <16 x i8> %3, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ; CHECK-DAG: nor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size nor_v16i8
}

define void @nor_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: nor_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = or <8 x i16> %1, %2
  %4 = xor <8 x i16> %3, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ; CHECK-DAG: nor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size nor_v8i16
}

define void @nor_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: nor_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = or <4 x i32> %1, %2
  %4 = xor <4 x i32> %3, <i32 -1, i32 -1, i32 -1, i32 -1>
  ; CHECK-DAG: nor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size nor_v4i32
}

define void @nor_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: nor_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = or <2 x i64> %1, %2
  %4 = xor <2 x i64> %3, <i64 -1, i64 -1>
  ; CHECK-DAG: nor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size nor_v2i64
}

define void @nor_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: nor_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = or <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = xor <16 x i8> %2, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ; CHECK-DAG: ori.b [[R4:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size nor_v16i8_i
}

define void @nor_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: nor_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = or <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = xor <8 x i16> %2, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: nor.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size nor_v8i16_i
}

define void @nor_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: nor_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = or <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = xor <4 x i32> %2, <i32 -1, i32 -1, i32 -1, i32 -1>
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: nor.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size nor_v4i32_i
}

define void @nor_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: nor_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = or <2 x i64> %1, <i64 1, i64 1>
  %3 = xor <2 x i64> %2, <i64 -1, i64 -1>
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: nor.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size nor_v2i64_i
}

define void @xor_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: xor_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = xor <16 x i8> %1, %2
  ; CHECK-DAG: xor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size xor_v16i8
}

define void @xor_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: xor_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = xor <8 x i16> %1, %2
  ; CHECK-DAG: xor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size xor_v8i16
}

define void @xor_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: xor_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = xor <4 x i32> %1, %2
  ; CHECK-DAG: xor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size xor_v4i32
}

define void @xor_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: xor_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = xor <2 x i64> %1, %2
  ; CHECK-DAG: xor.v [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size xor_v2i64
}

define void @xor_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: xor_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <16 x i8> %1, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ; CHECK-DAG: xori.b [[R4:\$w[0-9]+]], [[R1]], 3
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size xor_v16i8_i
}

define void @xor_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: xor_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <8 x i16> %1, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 3
  ; CHECK-DAG: xor.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size xor_v8i16_i
}

define void @xor_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: xor_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <4 x i32> %1, <i32 3, i32 3, i32 3, i32 3>
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 3
  ; CHECK-DAG: xor.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size xor_v4i32_i
}

define void @xor_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: xor_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <2 x i64> %1, <i64 3, i64 3>
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 3
  ; CHECK-DAG: xor.v [[R4:\$w[0-9]+]], [[R1]], [[R3]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size xor_v2i64_i
}

define void @sll_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: sll_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <16 x i8> %1, %2
  ; CHECK-DAG: sll.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size sll_v16i8
}

define void @sll_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: sll_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <8 x i16> %1, %2
  ; CHECK-DAG: sll.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size sll_v8i16
}

define void @sll_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: sll_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <4 x i32> %1, %2
  ; CHECK-DAG: sll.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size sll_v4i32
}

define void @sll_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: sll_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <2 x i64> %1, %2
  ; CHECK-DAG: sll.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size sll_v2i64
}

define void @sll_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: sll_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = shl <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: slli.b [[R4:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size sll_v16i8_i
}

define void @sll_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: sll_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = shl <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: slli.h [[R4:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size sll_v8i16_i
}

define void @sll_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: sll_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = shl <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: slli.w [[R4:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size sll_v4i32_i
}

define void @sll_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: sll_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = shl <2 x i64> %1, <i64 1, i64 1>
  ; CHECK-DAG: slli.d [[R4:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size sll_v2i64_i
}

define void @sra_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: sra_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = ashr <16 x i8> %1, %2
  ; CHECK-DAG: sra.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size sra_v16i8
}

define void @sra_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: sra_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = ashr <8 x i16> %1, %2
  ; CHECK-DAG: sra.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size sra_v8i16
}

define void @sra_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: sra_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = ashr <4 x i32> %1, %2
  ; CHECK-DAG: sra.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size sra_v4i32
}

define void @sra_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: sra_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = ashr <2 x i64> %1, %2
  ; CHECK-DAG: sra.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size sra_v2i64
}

define void @sra_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: sra_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = ashr <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: srai.b [[R4:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size sra_v16i8_i
}

define void @sra_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: sra_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = ashr <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: srai.h [[R4:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size sra_v8i16_i
}

define void @sra_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: sra_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = ashr <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: srai.w [[R4:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size sra_v4i32_i
}

define void @sra_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: sra_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = ashr <2 x i64> %1, <i64 1, i64 1>
  ; CHECK-DAG: srai.d [[R4:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size sra_v2i64_i
}

define void @srl_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: srl_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = lshr <16 x i8> %1, %2
  ; CHECK-DAG: srl.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size srl_v16i8
}

define void @srl_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: srl_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = lshr <8 x i16> %1, %2
  ; CHECK-DAG: srl.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size srl_v8i16
}

define void @srl_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: srl_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = lshr <4 x i32> %1, %2
  ; CHECK-DAG: srl.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size srl_v4i32
}

define void @srl_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: srl_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = lshr <2 x i64> %1, %2
  ; CHECK-DAG: srl.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size srl_v2i64
}

define void @srl_v16i8_i(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: srl_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = lshr <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: srli.b [[R4:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size srl_v16i8_i
}

define void @srl_v8i16_i(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: srl_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = lshr <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: srli.h [[R4:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size srl_v8i16_i
}

define void @srl_v4i32_i(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: srl_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = lshr <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: srli.w [[R4:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size srl_v4i32_i
}

define void @srl_v2i64_i(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: srl_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = lshr <2 x i64> %1, <i64 1, i64 1>
  ; CHECK-DAG: srli.d [[R4:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size srl_v2i64_i
}

define void @ctpop_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: ctpop_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <16 x i8> @llvm.ctpop.v16i8 (<16 x i8> %1)
  ; CHECK-DAG: pcnt.b [[R3:\$w[0-9]+]], [[R1]]
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ctpop_v16i8
}

define void @ctpop_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: ctpop_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <8 x i16> @llvm.ctpop.v8i16 (<8 x i16> %1)
  ; CHECK-DAG: pcnt.h [[R3:\$w[0-9]+]], [[R1]]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ctpop_v8i16
}

define void @ctpop_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: ctpop_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <4 x i32> @llvm.ctpop.v4i32 (<4 x i32> %1)
  ; CHECK-DAG: pcnt.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ctpop_v4i32
}

define void @ctpop_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: ctpop_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <2 x i64> @llvm.ctpop.v2i64 (<2 x i64> %1)
  ; CHECK-DAG: pcnt.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ctpop_v2i64
}

define void @ctlz_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: ctlz_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <16 x i8> @llvm.ctlz.v16i8 (<16 x i8> %1)
  ; CHECK-DAG: nlzc.b [[R3:\$w[0-9]+]], [[R1]]
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ctlz_v16i8
}

define void @ctlz_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: ctlz_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <8 x i16> @llvm.ctlz.v8i16 (<8 x i16> %1)
  ; CHECK-DAG: nlzc.h [[R3:\$w[0-9]+]], [[R1]]
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ctlz_v8i16
}

define void @ctlz_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: ctlz_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <4 x i32> @llvm.ctlz.v4i32 (<4 x i32> %1)
  ; CHECK-DAG: nlzc.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ctlz_v4i32
}

define void @ctlz_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: ctlz_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <2 x i64> @llvm.ctlz.v2i64 (<2 x i64> %1)
  ; CHECK-DAG: nlzc.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ctlz_v2i64
}

define void @bsel_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b, <16 x i8>* %m) nounwind {
  ; CHECK: bsel_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = load <16 x i8>, <16 x i8>* %m
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0($7)
  %4 = xor <16 x i8> %3, <i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1>
  %5 = and <16 x i8> %1, %3
  %6 = and <16 x i8> %2, %4
  %7 = or <16 x i8> %5, %6
  ; bmnz is the same operation
  ; (vselect Mask, IfSet, IfClr) -> (BMNZ IfClr, IfSet, Mask)
  ; CHECK-DAG: bmnz.v [[R2]], [[R1]], [[R3]]
  store <16 x i8> %7, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R2]], 0($4)

  ret void
  ; CHECK: .size bsel_v16i8
}

define void @bsel_v16i8_i(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %m) nounwind {
  ; CHECK: bsel_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %m
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0($6)
  %3 = xor <16 x i8> %2, <i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1>
  %4 = and <16 x i8> %1, %3
  %5 = and <16 x i8> <i8 6, i8 6, i8 6, i8 6,
                      i8 6, i8 6, i8 6, i8 6,
                      i8 6, i8 6, i8 6, i8 6,
                      i8 6, i8 6, i8 6, i8 6>, %2
  %6 = or <16 x i8> %4, %5
  ; CHECK-DAG: bseli.b [[R3]], [[R1]], 6
  store <16 x i8> %6, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bsel_v16i8_i
}

define void @bsel_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: bsel_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = and <8 x i16> %1, <i16 6, i16 6, i16 6, i16 6,
                          i16 6, i16 6, i16 6, i16 6>
  %4 = and <8 x i16> %2, <i16 65529, i16 65529, i16 65529, i16 65529,
                          i16 65529, i16 65529, i16 65529, i16 65529>
  %5 = or <8 x i16> %3, %4
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 6
  ; CHECK-DAG: bsel.v [[R3]], [[R2]], [[R1]]
  store <8 x i16> %5, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size bsel_v8i16
}

define void @bsel_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: bsel_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = and <4 x i32> %1, <i32 6, i32 6, i32 6, i32 6>
  %4 = and <4 x i32> %2, <i32 4294967289, i32 4294967289, i32 4294967289, i32 4294967289>
  %5 = or <4 x i32> %3, %4
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 6
  ; CHECK-DAG: bsel.v [[R3]], [[R2]], [[R1]]
  store <4 x i32> %5, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size bsel_v4i32
}

define void @bsel_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: bsel_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = and <2 x i64> %1, <i64 6, i64 6>
  %4 = and <2 x i64> %2, <i64 18446744073709551609, i64 18446744073709551609>
  %5 = or <2 x i64> %3, %4
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 6
  ; CHECK-DAG: bsel.v [[R3]], [[R2]], [[R1]]
  store <2 x i64> %5, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size bsel_v2i64
}

define void @binsl_v16i8_i(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: binsl_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = and <16 x i8> %1, <i8 192, i8 192, i8 192, i8 192,
                          i8 192, i8 192, i8 192, i8 192,
                          i8 192, i8 192, i8 192, i8 192,
                          i8 192, i8 192, i8 192, i8 192>
  %4 = and <16 x i8> %2, <i8 63, i8 63, i8 63, i8 63,
                          i8 63, i8 63, i8 63, i8 63,
                          i8 63, i8 63, i8 63, i8 63,
                          i8 63, i8 63, i8 63, i8 63>
  %5 = or <16 x i8> %3, %4
  ; CHECK-DAG: binsli.b [[R2]], [[R1]], 1
  store <16 x i8> %5, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R2]], 0($4)

  ret void
  ; CHECK: .size binsl_v16i8_i
}

define void @binsl_v8i16_i(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: binsl_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = and <8 x i16> %1, <i16 49152, i16 49152, i16 49152, i16 49152,
                          i16 49152, i16 49152, i16 49152, i16 49152>
  %4 = and <8 x i16> %2, <i16 16383, i16 16383, i16 16383, i16 16383,
                          i16 16383, i16 16383, i16 16383, i16 16383>
  %5 = or <8 x i16> %3, %4
  ; CHECK-DAG: binsli.h [[R2]], [[R1]], 1
  store <8 x i16> %5, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R2]], 0($4)

  ret void
  ; CHECK: .size binsl_v8i16_i
}

define void @binsl_v4i32_i(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: binsl_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = and <4 x i32> %1, <i32 3221225472, i32 3221225472, i32 3221225472, i32 3221225472>
  %4 = and <4 x i32> %2, <i32 1073741823, i32 1073741823, i32 1073741823, i32 1073741823>
  %5 = or <4 x i32> %3, %4
  ; CHECK-DAG: binsli.w [[R2]], [[R1]], 1
  store <4 x i32> %5, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R2]], 0($4)

  ret void
  ; CHECK: .size binsl_v4i32_i
}

define void @binsl_v2i64_i(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: binsl_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = and <2 x i64> %1, <i64 18446744073709551608, i64 18446744073709551608>
  %4 = and <2 x i64> %2, <i64 7, i64 7>
  %5 = or <2 x i64> %3, %4
  ; TODO: We use a particularly wide mask here to work around a legalization
  ;       issue. If the mask doesn't fit within a 10-bit immediate, it gets
  ;       legalized into a constant pool. We should add a test to cover the
  ;       other cases once they correctly select binsli.d.
  ; CHECK-DAG: binsli.d [[R2]], [[R1]], 60
  store <2 x i64> %5, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R2]], 0($4)

  ret void
  ; CHECK: .size binsl_v2i64_i
}

define void @binsr_v16i8_i(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: binsr_v16i8_i:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = and <16 x i8> %1, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3,
                          i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  %4 = and <16 x i8> %2, <i8 252, i8 252, i8 252, i8 252,
                          i8 252, i8 252, i8 252, i8 252,
                          i8 252, i8 252, i8 252, i8 252,
                          i8 252, i8 252, i8 252, i8 252>
  %5 = or <16 x i8> %3, %4
  ; CHECK-DAG: binsri.b [[R2]], [[R1]], 1
  store <16 x i8> %5, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R2]], 0($4)

  ret void
  ; CHECK: .size binsr_v16i8_i
}

define void @binsr_v8i16_i(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: binsr_v8i16_i:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = and <8 x i16> %1, <i16 3, i16 3, i16 3, i16 3,
                          i16 3, i16 3, i16 3, i16 3>
  %4 = and <8 x i16> %2, <i16 65532, i16 65532, i16 65532, i16 65532,
                          i16 65532, i16 65532, i16 65532, i16 65532>
  %5 = or <8 x i16> %3, %4
  ; CHECK-DAG: binsri.h [[R2]], [[R1]], 1
  store <8 x i16> %5, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R2]], 0($4)

  ret void
  ; CHECK: .size binsr_v8i16_i
}

define void @binsr_v4i32_i(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: binsr_v4i32_i:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = and <4 x i32> %1, <i32 3, i32 3, i32 3, i32 3>
  %4 = and <4 x i32> %2, <i32 4294967292, i32 4294967292, i32 4294967292, i32 4294967292>
  %5 = or <4 x i32> %3, %4
  ; CHECK-DAG: binsri.w [[R2]], [[R1]], 1
  store <4 x i32> %5, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R2]], 0($4)

  ret void
  ; CHECK: .size binsr_v4i32_i
}

define void @binsr_v2i64_i(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: binsr_v2i64_i:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = and <2 x i64> %1, <i64 3, i64 3>
  %4 = and <2 x i64> %2, <i64 18446744073709551612, i64 18446744073709551612>
  %5 = or <2 x i64> %3, %4
  ; CHECK-DAG: binsri.d [[R2]], [[R1]], 1
  store <2 x i64> %5, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R2]], 0($4)

  ret void
  ; CHECK: .size binsr_v2i64_i
}

define void @bclr_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: bclr_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, %2
  %4 = xor <16 x i8> %3, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %5 = and <16 x i8> %1, %4
  ; CHECK-DAG: bclr.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %5, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bclr_v16i8
}

define void @bclr_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: bclr_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, %2
  %4 = xor <8 x i16> %3, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %5 = and <8 x i16> %1, %4
  ; CHECK-DAG: bclr.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %5, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size bclr_v8i16
}

define void @bclr_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: bclr_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <4 x i32> <i32 1, i32 1, i32 1, i32 1>, %2
  %4 = xor <4 x i32> %3, <i32 -1, i32 -1, i32 -1, i32 -1>
  %5 = and <4 x i32> %1, %4
  ; CHECK-DAG: bclr.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %5, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size bclr_v4i32
}

define void @bclr_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: bclr_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <2 x i64> <i64 1, i64 1>, %2
  %4 = xor <2 x i64> %3, <i64 -1, i64 -1>
  %5 = and <2 x i64> %1, %4
  ; CHECK-DAG: bclr.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %5, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size bclr_v2i64
}

define void @bset_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: bset_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, %2
  %4 = or <16 x i8> %1, %3
  ; CHECK-DAG: bset.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bset_v16i8
}

define void @bset_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: bset_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, %2
  %4 = or <8 x i16> %1, %3
  ; CHECK-DAG: bset.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size bset_v8i16
}

define void @bset_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: bset_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <4 x i32> <i32 1, i32 1, i32 1, i32 1>, %2
  %4 = or <4 x i32> %1, %3
  ; CHECK-DAG: bset.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size bset_v4i32
}

define void @bset_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: bset_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <2 x i64> <i64 1, i64 1>, %2
  %4 = or <2 x i64> %1, %3
  ; CHECK-DAG: bset.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size bset_v2i64
}

define void @bneg_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: bneg_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, %2
  %4 = xor <16 x i8> %1, %3
  ; CHECK-DAG: bneg.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bneg_v16i8
}

define void @bneg_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: bneg_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, %2
  %4 = xor <8 x i16> %1, %3
  ; CHECK-DAG: bneg.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size bneg_v8i16
}

define void @bneg_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: bneg_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <4 x i32> <i32 1, i32 1, i32 1, i32 1>, %2
  %4 = xor <4 x i32> %1, %3
  ; CHECK-DAG: bneg.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size bneg_v4i32
}

define void @bneg_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: bneg_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = shl <2 x i64> <i64 1, i64 1>, %2
  %4 = xor <2 x i64> %1, %3
  ; CHECK-DAG: bneg.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size bneg_v2i64
}

define void @bclri_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: bclri_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <16 x i8> <i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8, i8  8>,
                     <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %3 = and <16 x i8> %1, %2
  ; bclri.b and andi.b are exactly equivalent.
  ; CHECK-DAG: andi.b [[R3:\$w[0-9]+]], [[R1]], 247
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bclri_v16i8
}

define void @bclri_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: bclri_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <8 x i16> <i16  8, i16  8, i16  8, i16  8, i16  8, i16  8, i16  8, i16  8>,
                     <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %3 = and <8 x i16> %1, %2
  ; CHECK-DAG: bclri.h [[R3:\$w[0-9]+]], [[R1]], 3
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size bclri_v8i16
}

define void @bclri_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: bclri_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <4 x i32> <i32  8, i32  8, i32  8, i32  8>,
                     <i32 -1, i32 -1, i32 -1, i32 -1>
  %3 = and <4 x i32> %1, %2
  ; CHECK-DAG: bclri.w [[R3:\$w[0-9]+]], [[R1]], 3
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size bclri_v4i32
}

define void @bclri_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: bclri_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <2 x i64> <i64  8, i64  8>,
                     <i64 -1, i64 -1>
  %3 = and <2 x i64> %1, %2
  ; CHECK-DAG: bclri.d [[R3:\$w[0-9]+]], [[R1]], 3
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size bclri_v2i64
}

define void @bseti_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: bseti_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = or <16 x i8> %1, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
  ; CHECK-DAG: bseti.b [[R3:\$w[0-9]+]], [[R1]], 3
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bseti_v16i8
}

define void @bseti_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: bseti_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = or <8 x i16> %1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ; CHECK-DAG: bseti.h [[R3:\$w[0-9]+]], [[R1]], 3
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size bseti_v8i16
}

define void @bseti_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: bseti_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = or <4 x i32> %1, <i32 8, i32 8, i32 8, i32 8>
  ; CHECK-DAG: bseti.w [[R3:\$w[0-9]+]], [[R1]], 3
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size bseti_v4i32
}

define void @bseti_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: bseti_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = or <2 x i64> %1, <i64 8, i64 8>
  ; CHECK-DAG: bseti.d [[R3:\$w[0-9]+]], [[R1]], 3
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size bseti_v2i64
}

define void @bnegi_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: bnegi_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <16 x i8> %1, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
  ; CHECK-DAG: bnegi.b [[R3:\$w[0-9]+]], [[R1]], 3
  store <16 x i8> %2, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bnegi_v16i8
}

define void @bnegi_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: bnegi_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <8 x i16> %1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ; CHECK-DAG: bnegi.h [[R3:\$w[0-9]+]], [[R1]], 3
  store <8 x i16> %2, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size bnegi_v8i16
}

define void @bnegi_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: bnegi_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <4 x i32> %1, <i32 8, i32 8, i32 8, i32 8>
  ; CHECK-DAG: bnegi.w [[R3:\$w[0-9]+]], [[R1]], 3
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size bnegi_v4i32
}

define void @bnegi_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: bnegi_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = xor <2 x i64> %1, <i64 8, i64 8>
  ; CHECK-DAG: bnegi.d [[R3:\$w[0-9]+]], [[R1]], 3
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size bnegi_v2i64
}

declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %val)
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %val)
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %val)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %val)
declare <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %val)
declare <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %val)
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %val)
declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %val)
