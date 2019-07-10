; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

define void @ceq_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: ceq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: ceq.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v16i8
}

define void @ceq_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: ceq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: ceq.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v8i16
}

define void @ceq_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: ceq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: ceq.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v4i32
}

define void @ceq_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: ceq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp eq <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: ceq.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ceq_v2i64
}

define void @cle_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: cle_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: cle_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v16i8
}

define void @cle_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: cle_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: cle_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v8i16
}

define void @cle_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: cle_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: cle_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v4i32
}

define void @cle_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: cle_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: cle_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_s_v2i64
}

define void @cle_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: cle_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: cle_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v16i8
}

define void @cle_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: cle_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: cle_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v8i16
}

define void @cle_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: cle_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: cle_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v4i32
}

define void @cle_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: cle_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: cle_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size cle_u_v2i64
}

define void @clt_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: clt_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: clt_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v16i8
}

define void @clt_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: clt_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: clt_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v8i16
}

define void @clt_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: clt_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: clt_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v4i32
}

define void @clt_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: clt_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: clt_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_s_v2i64
}

define void @clt_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: clt_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: clt_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v16i8
}

define void @clt_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: clt_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: clt_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v8i16
}

define void @clt_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: clt_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: clt_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v4i32
}

define void @clt_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: clt_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: clt_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clt_u_v2i64
}

; There is no != comparison, but test it anyway since we've had legalizer
; issues in this area.
define void @cne_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: cne_v16i8:
  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ne <16 x i8> %1, %2
  %4 = sext <16 x i1> %3 to <16 x i8>
  ; CHECK-DAG: ceq.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  ; CHECK-DAG: xori.b [[R3]], [[R3]], 255
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size cne_v16i8
}

; There is no != comparison, but test it anyway since we've had legalizer
; issues in this area.
define void @cne_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: cne_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ne <8 x i16> %1, %2
  %4 = sext <8 x i1> %3 to <8 x i16>
  ; CHECK-DAG: ceq.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  ; TODO: This should be an 'xori.b [[R3]], [[R3]], 255' but thats an optimisation issue
  ; CHECK-DAG: ldi.b [[R4:\$w[0-9]+]], -1
  ; CHECK-DAG: xor.v [[R3]], [[R3]], [[R4]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size cne_v8i16
}

; There is no != comparison, but test it anyway since we've had legalizer
; issues in this area.
define void @cne_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: cne_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ne <4 x i32> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  ; CHECK-DAG: ceq.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  ; TODO: This should be an 'xori.b [[R3]], [[R3]], 255' but thats an optimisation issue
  ; CHECK-DAG: ldi.b [[R4:\$w[0-9]+]], -1
  ; CHECK-DAG: xor.v [[R3]], [[R3]], [[R4]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size cne_v4i32
}

; There is no != comparison, but test it anyway since we've had legalizer
; issues in this area.
define void @cne_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: cne_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ne <2 x i64> %1, %2
  %4 = sext <2 x i1> %3 to <2 x i64>
  ; CHECK-DAG: ceq.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  ; TODO: This should be an 'xori.b [[R3]], [[R3]], 255' but thats an optimisation issue
  ; CHECK-DAG: ldi.b [[R4:\$w[0-9]+]], -1
  ; CHECK-DAG: xor.v [[R3]], [[R3]], [[R4]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size cne_v2i64
}

define void @ceqi_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: ceqi_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: ceqi.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v16i8
}

define void @ceqi_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: ceqi_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: ceqi.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v8i16
}

define void @ceqi_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: ceqi_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: ceqi.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v4i32
}

define void @ceqi_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: ceqi_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp eq <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: ceqi.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ceqi_v2i64
}

define void @clei_s_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clei_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clei_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v16i8
}

define void @clei_s_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clei_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clei_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v8i16
}

define void @clei_s_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clei_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clei_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v4i32
}

define void @clei_s_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clei_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clei_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_s_v2i64
}

define void @clei_u_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clei_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clei_u.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v16i8
}

define void @clei_u_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clei_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clei_u.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v8i16
}

define void @clei_u_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clei_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clei_u.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v4i32
}

define void @clei_u_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clei_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clei_u.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clei_u_v2i64
}

define void @clti_s_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clti_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clti_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v16i8
}

define void @clti_s_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clti_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clti_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v8i16
}

define void @clti_s_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clti_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clti_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v4i32
}

define void @clti_s_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clti_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <2 x i64> %1, <i64 1, i64 1>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clti_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_s_v2i64
}

define void @clti_u_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: clti_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <16 x i8> %1, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  %3 = sext <16 x i1> %2 to <16 x i8>
  ; CHECK-DAG: clti_u.b [[R3:\$w[0-9]+]], [[R1]], 2
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v16i8
}

define void @clti_u_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: clti_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <8 x i16> %1, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ; CHECK-DAG: clti_u.h [[R3:\$w[0-9]+]], [[R1]], 2
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v8i16
}

define void @clti_u_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: clti_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <4 x i32> %1, <i32 2, i32 2, i32 2, i32 2>
  %3 = sext <4 x i1> %2 to <4 x i32>
  ; CHECK-DAG: clti_u.w [[R3:\$w[0-9]+]], [[R1]], 2
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v4i32
}

define void @clti_u_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: clti_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <2 x i64> %1, <i64 2, i64 2>
  %3 = sext <2 x i1> %2 to <2 x i64>
  ; CHECK-DAG: clti_u.d [[R3:\$w[0-9]+]], [[R1]], 2
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size clti_u_v2i64
}

define void @bsel_s_v16i8(<16 x i8>* %d, <16 x i8>* %a, <16 x i8>* %b,
                        <16 x i8>* %c) nounwind {
  ; CHECK: bsel_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = load <16 x i8>, <16 x i8>* %c
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp sgt <16 x i8> %1, %2
  ; CHECK-DAG: clt_s.b [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <16 x i1> %4, <16 x i8> %1, <16 x i8> %3
  ; bmnz.v is the same operation
  ; CHECK-DAG: bmnz.v [[R3]], [[R1]], [[R4]]
  store <16 x i8> %5, <16 x i8>* %d
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bsel_s_v16i8
}

define void @bsel_s_v8i16(<8 x i16>* %d, <8 x i16>* %a, <8 x i16>* %b,
                        <8 x i16>* %c) nounwind {
  ; CHECK: bsel_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = load <8 x i16>, <8 x i16>* %c
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp sgt <8 x i16> %1, %2
  ; CHECK-DAG: clt_s.h [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <8 x i1> %4, <8 x i16> %1, <8 x i16> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <8 x i16> %5, <8 x i16>* %d
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_s_v8i16
}

define void @bsel_s_v4i32(<4 x i32>* %d, <4 x i32>* %a, <4 x i32>* %b,
                        <4 x i32>* %c) nounwind {
  ; CHECK: bsel_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = load <4 x i32>, <4 x i32>* %c
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp sgt <4 x i32> %1, %2
  ; CHECK-DAG: clt_s.w [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <4 x i1> %4, <4 x i32> %1, <4 x i32> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <4 x i32> %5, <4 x i32>* %d
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_s_v4i32
}

define void @bsel_s_v2i64(<2 x i64>* %d, <2 x i64>* %a, <2 x i64>* %b,
                        <2 x i64>* %c) nounwind {
  ; CHECK: bsel_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = load <2 x i64>, <2 x i64>* %c
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp sgt <2 x i64> %1, %2
  ; CHECK-DAG: clt_s.d [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <2 x i1> %4, <2 x i64> %1, <2 x i64> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <2 x i64> %5, <2 x i64>* %d
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_s_v2i64
}

define void @bsel_u_v16i8(<16 x i8>* %d, <16 x i8>* %a, <16 x i8>* %b,
                        <16 x i8>* %c) nounwind {
  ; CHECK: bsel_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = load <16 x i8>, <16 x i8>* %c
  ; CHECK-DAG: ld.b [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp ugt <16 x i8> %1, %2
  ; CHECK-DAG: clt_u.b [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <16 x i1> %4, <16 x i8> %1, <16 x i8> %3
  ; bmnz.v is the same operation
  ; CHECK-DAG: bmnz.v [[R3]], [[R1]], [[R4]]
  store <16 x i8> %5, <16 x i8>* %d
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size bsel_u_v16i8
}

define void @bsel_u_v8i16(<8 x i16>* %d, <8 x i16>* %a, <8 x i16>* %b,
                        <8 x i16>* %c) nounwind {
  ; CHECK: bsel_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = load <8 x i16>, <8 x i16>* %c
  ; CHECK-DAG: ld.h [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp ugt <8 x i16> %1, %2
  ; CHECK-DAG: clt_u.h [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <8 x i1> %4, <8 x i16> %1, <8 x i16> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <8 x i16> %5, <8 x i16>* %d
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_u_v8i16
}

define void @bsel_u_v4i32(<4 x i32>* %d, <4 x i32>* %a, <4 x i32>* %b,
                        <4 x i32>* %c) nounwind {
  ; CHECK: bsel_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = load <4 x i32>, <4 x i32>* %c
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp ugt <4 x i32> %1, %2
  ; CHECK-DAG: clt_u.w [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <4 x i1> %4, <4 x i32> %1, <4 x i32> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <4 x i32> %5, <4 x i32>* %d
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_u_v4i32
}

define void @bsel_u_v2i64(<2 x i64>* %d, <2 x i64>* %a, <2 x i64>* %b,
                        <2 x i64>* %c) nounwind {
  ; CHECK: bsel_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = load <2 x i64>, <2 x i64>* %c
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0($7)
  %4 = icmp ugt <2 x i64> %1, %2
  ; CHECK-DAG: clt_u.d [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %5 = select <2 x i1> %4, <2 x i64> %1, <2 x i64> %3
  ; Note that IfSet and IfClr are swapped since the condition is inverted
  ; CHECK-DAG: bsel.v [[R4]], [[R3]], [[R1]]
  store <2 x i64> %5, <2 x i64>* %d
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size bsel_u_v2i64
}

define void @bseli_s_v16i8(<16 x i8>* %d, <16 x i8>* %a, <16 x i8>* %b,
                        <16 x i8>* %c) nounwind {
  ; CHECK: bseli_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <16 x i8> %1, %2
  ; CHECK-DAG: clt_s.b [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <16 x i1> %3, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8> %1
  ; CHECK-DAG: bseli.b [[R4]], [[R1]], 1
  store <16 x i8> %4, <16 x i8>* %d
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_s_v16i8
}

define void @bseli_s_v8i16(<8 x i16>* %d, <8 x i16>* %a, <8 x i16>* %b,
                        <8 x i16>* %c) nounwind {
  ; CHECK: bseli_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <8 x i16> %1, %2
  ; CHECK-DAG: clt_s.h [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <8 x i1> %3, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16> %1
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: bsel.v [[R4]], [[R1]], [[R3]]
  store <8 x i16> %4, <8 x i16>* %d
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_s_v8i16
}

define void @bseli_s_v4i32(<4 x i32>* %d, <4 x i32>* %a, <4 x i32>* %b,
                        <4 x i32>* %c) nounwind {
  ; CHECK: bseli_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <4 x i32> %1, %2
  ; CHECK-DAG: clt_s.w [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <4 x i1> %3, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> %1
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: bsel.v [[R4]], [[R1]], [[R3]]
  store <4 x i32> %4, <4 x i32>* %d
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_s_v4i32
}

define void @bseli_s_v2i64(<2 x i64>* %d, <2 x i64>* %a, <2 x i64>* %b,
                        <2 x i64>* %c) nounwind {
  ; CHECK: bseli_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <2 x i64> %1, %2
  ; CHECK-DAG: clt_s.d [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <2 x i1> %3, <2 x i64> <i64 1, i64 1>, <2 x i64> %1
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: bsel.v [[R4]], [[R1]], [[R3]]
  store <2 x i64> %4, <2 x i64>* %d
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_s_v2i64
}

define void @bseli_u_v16i8(<16 x i8>* %d, <16 x i8>* %a, <16 x i8>* %b,
                        <16 x i8>* %c) nounwind {
  ; CHECK: bseli_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <16 x i8> %1, %2
  ; CHECK-DAG: clt_u.b [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <16 x i1> %3, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8> %1
  ; CHECK-DAG: bseli.b [[R4]], [[R1]], 1
  store <16 x i8> %4, <16 x i8>* %d
  ; CHECK-DAG: st.b [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_u_v16i8
}

define void @bseli_u_v8i16(<8 x i16>* %d, <8 x i16>* %a, <8 x i16>* %b,
                        <8 x i16>* %c) nounwind {
  ; CHECK: bseli_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <8 x i16> %1, %2
  ; CHECK-DAG: clt_u.h [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <8 x i1> %3, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16> %1
  ; CHECK-DAG: ldi.h [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: bsel.v [[R4]], [[R1]], [[R3]]
  store <8 x i16> %4, <8 x i16>* %d
  ; CHECK-DAG: st.h [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_u_v8i16
}

define void @bseli_u_v4i32(<4 x i32>* %d, <4 x i32>* %a, <4 x i32>* %b,
                        <4 x i32>* %c) nounwind {
  ; CHECK: bseli_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <4 x i32> %1, %2
  ; CHECK-DAG: clt_u.w [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <4 x i1> %3, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> %1
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: bsel.v [[R4]], [[R1]], [[R3]]
  store <4 x i32> %4, <4 x i32>* %d
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_u_v4i32
}

define void @bseli_u_v2i64(<2 x i64>* %d, <2 x i64>* %a, <2 x i64>* %b,
                        <2 x i64>* %c) nounwind {
  ; CHECK: bseli_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <2 x i64> %1, %2
  ; CHECK-DAG: clt_u.d [[R4:\$w[0-9]+]], [[R2]], [[R1]]
  %4 = select <2 x i1> %3, <2 x i64> <i64 1, i64 1>, <2 x i64> %1
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: bsel.v [[R4]], [[R1]], [[R3]]
  store <2 x i64> %4, <2 x i64>* %d
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size bseli_u_v2i64
}

define void @max_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: max_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: max_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_v16i8
}

define void @max_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: max_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: max_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_v8i16
}

define void @max_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: max_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: max_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_v4i32
}

define void @max_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: max_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sgt <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: max_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_v2i64
}

define void @max_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: max_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: max_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_v16i8
}

define void @max_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: max_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: max_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_v8i16
}

define void @max_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: max_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: max_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_v4i32
}

define void @max_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: max_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ugt <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: max_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_v2i64
}

define void @max_s_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: max_s_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sge <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: max_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_eq_v16i8
}

define void @max_s_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: max_s_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sge <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: max_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_eq_v8i16
}

define void @max_s_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: max_s_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sge <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: max_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_eq_v4i32
}

define void @max_s_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: max_s_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sge <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: max_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size max_s_eq_v2i64
}

define void @max_u_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: max_u_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp uge <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: max_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_eq_v16i8
}

define void @max_u_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: max_u_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp uge <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: max_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_eq_v8i16
}

define void @max_u_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: max_u_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp uge <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: max_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_eq_v4i32
}

define void @max_u_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: max_u_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp uge <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: max_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size max_u_eq_v2i64
}

define void @maxi_s_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: maxi_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sgt <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: maxi_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_v16i8
}

define void @maxi_s_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: maxi_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sgt <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: maxi_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_v8i16
}

define void @maxi_s_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: maxi_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sgt <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: maxi_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_v4i32
}

define void @maxi_s_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: maxi_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sgt <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: maxi_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_v2i64
}

define void @maxi_u_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: maxi_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ugt <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: maxi_u.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_v16i8
}

define void @maxi_u_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: maxi_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ugt <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: maxi_u.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_v8i16
}

define void @maxi_u_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: maxi_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ugt <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: maxi_u.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_v4i32
}

define void @maxi_u_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: maxi_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ugt <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: maxi_u.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_v2i64
}

define void @maxi_s_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: maxi_s_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sge <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: maxi_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_eq_v16i8
}

define void @maxi_s_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: maxi_s_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sge <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: maxi_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_eq_v8i16
}

define void @maxi_s_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: maxi_s_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sge <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: maxi_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_eq_v4i32
}

define void @maxi_s_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: maxi_s_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sge <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: maxi_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_s_eq_v2i64
}

define void @maxi_u_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: maxi_u_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp uge <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: maxi_u.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_eq_v16i8
}

define void @maxi_u_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: maxi_u_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp uge <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: maxi_u.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_eq_v8i16
}

define void @maxi_u_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: maxi_u_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp uge <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: maxi_u.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_eq_v4i32
}

define void @maxi_u_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: maxi_u_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp uge <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: maxi_u.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size maxi_u_eq_v2i64
}

define void @min_s_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: min_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: min_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_v16i8
}

define void @min_s_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: min_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: min_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_v8i16
}

define void @min_s_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: min_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: min_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_v4i32
}

define void @min_s_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: min_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp slt <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: min_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_v2i64
}

define void @min_u_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: min_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: min_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_v16i8
}

define void @min_u_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: min_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: min_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_v8i16
}

define void @min_u_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: min_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: min_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_v4i32
}

define void @min_u_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: min_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ult <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: min_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_v2i64
}

define void @min_s_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: min_s_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: min_s.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_eq_v16i8
}

define void @min_s_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: min_s_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: min_s.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_eq_v8i16
}

define void @min_s_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: min_s_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: min_s.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_eq_v4i32
}

define void @min_s_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: min_s_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp sle <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: min_s.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size min_s_eq_v2i64
}

define void @min_u_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a, <16 x i8>* %b) nounwind {
  ; CHECK: min_u_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = load <16 x i8>, <16 x i8>* %b
  ; CHECK-DAG: ld.b [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <16 x i8> %1, %2
  %4 = select <16 x i1> %3, <16 x i8> %1, <16 x i8> %2
  ; CHECK-DAG: min_u.b [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <16 x i8> %4, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_eq_v16i8
}

define void @min_u_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a, <8 x i16>* %b) nounwind {
  ; CHECK: min_u_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = load <8 x i16>, <8 x i16>* %b
  ; CHECK-DAG: ld.h [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <8 x i16> %1, %2
  %4 = select <8 x i1> %3, <8 x i16> %1, <8 x i16> %2
  ; CHECK-DAG: min_u.h [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <8 x i16> %4, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_eq_v8i16
}

define void @min_u_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a, <4 x i32>* %b) nounwind {
  ; CHECK: min_u_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x i32>, <4 x i32>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <4 x i32> %1, %2
  %4 = select <4 x i1> %3, <4 x i32> %1, <4 x i32> %2
  ; CHECK-DAG: min_u.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x i32> %4, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_eq_v4i32
}

define void @min_u_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a, <2 x i64>* %b) nounwind {
  ; CHECK: min_u_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x i64>, <2 x i64>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = icmp ule <2 x i64> %1, %2
  %4 = select <2 x i1> %3, <2 x i64> %1, <2 x i64> %2
  ; CHECK-DAG: min_u.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x i64> %4, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size min_u_eq_v2i64
}

define void @mini_s_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: mini_s_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: mini_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_v16i8
}

define void @mini_s_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: mini_s_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: mini_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_v8i16
}

define void @mini_s_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: mini_s_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: mini_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_v4i32
}

define void @mini_s_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: mini_s_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp slt <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: mini_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_v2i64
}

define void @mini_u_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: mini_u_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: mini_u.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_v16i8
}

define void @mini_u_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: mini_u_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: mini_u.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_v8i16
}

define void @mini_u_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: mini_u_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: mini_u.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_v4i32
}

define void @mini_u_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: mini_u_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ult <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: mini_u.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_v2i64
}

define void @mini_s_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: mini_s_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: mini_s.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_eq_v16i8
}

define void @mini_s_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: mini_s_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: mini_s.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_eq_v8i16
}

define void @mini_s_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: mini_s_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: mini_s.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_eq_v4i32
}

define void @mini_s_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: mini_s_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp sle <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: mini_s.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_s_eq_v2i64
}

define void @mini_u_eq_v16i8(<16 x i8>* %c, <16 x i8>* %a) nounwind {
  ; CHECK: mini_u_eq_v16i8:

  %1 = load <16 x i8>, <16 x i8>* %a
  ; CHECK-DAG: ld.b [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <16 x i8> %1, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %3 = select <16 x i1> %2, <16 x i8> %1, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ; CHECK-DAG: mini_u.b [[R3:\$w[0-9]+]], [[R1]], 1
  store <16 x i8> %3, <16 x i8>* %c
  ; CHECK-DAG: st.b [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_eq_v16i8
}

define void @mini_u_eq_v8i16(<8 x i16>* %c, <8 x i16>* %a) nounwind {
  ; CHECK: mini_u_eq_v8i16:

  %1 = load <8 x i16>, <8 x i16>* %a
  ; CHECK-DAG: ld.h [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <8 x i16> %1, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %3 = select <8 x i1> %2, <8 x i16> %1, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ; CHECK-DAG: mini_u.h [[R3:\$w[0-9]+]], [[R1]], 1
  store <8 x i16> %3, <8 x i16>* %c
  ; CHECK-DAG: st.h [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_eq_v8i16
}

define void @mini_u_eq_v4i32(<4 x i32>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: mini_u_eq_v4i32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  %3 = select <4 x i1> %2, <4 x i32> %1, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ; CHECK-DAG: mini_u.w [[R3:\$w[0-9]+]], [[R1]], 1
  store <4 x i32> %3, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_eq_v4i32
}

define void @mini_u_eq_v2i64(<2 x i64>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: mini_u_eq_v2i64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = icmp ule <2 x i64> %1, <i64 1, i64 1>
  %3 = select <2 x i1> %2, <2 x i64> %1, <2 x i64> <i64 1, i64 1>
  ; CHECK-DAG: mini_u.d [[R3:\$w[0-9]+]], [[R1]], 1
  store <2 x i64> %3, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mini_u_eq_v2i64
}
