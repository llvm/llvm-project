; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

define void @add_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: add_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fadd <4 x float> %1, %2
  ; CHECK-DAG: fadd.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v4f32
}

define void @add_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: add_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fadd <2 x double> %1, %2
  ; CHECK-DAG: fadd.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size add_v2f64
}

define void @sub_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: sub_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fsub <4 x float> %1, %2
  ; CHECK-DAG: fsub.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v4f32
}

define void @sub_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: sub_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fsub <2 x double> %1, %2
  ; CHECK-DAG: fsub.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size sub_v2f64
}

define void @mul_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: mul_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fmul <4 x float> %1, %2
  ; CHECK-DAG: fmul.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v4f32
}

define void @mul_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: mul_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fmul <2 x double> %1, %2
  ; CHECK-DAG: fmul.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size mul_v2f64
}

define void @fma_v4f32(<4 x float>* %d, <4 x float>* %a, <4 x float>* %b,
                       <4 x float>* %c) nounwind {
  ; CHECK: fma_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = load <4 x float>, <4 x float>* %c
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0($7)
  %4 = tail call <4 x float> @llvm.fma.v4f32 (<4 x float> %1, <4 x float> %2,
                                              <4 x float> %3)
  ; CHECK-DAG: fmadd.w [[R1]], [[R2]], [[R3]]
  store <4 x float> %4, <4 x float>* %d
  ; CHECK-DAG: st.w [[R1]], 0($4)

  ret void
  ; CHECK: .size fma_v4f32
}

define void @fma_v2f64(<2 x double>* %d, <2 x double>* %a, <2 x double>* %b,
                       <2 x double>* %c) nounwind {
  ; CHECK: fma_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = load <2 x double>, <2 x double>* %c
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0($7)
  %4 = tail call <2 x double> @llvm.fma.v2f64 (<2 x double> %1, <2 x double> %2,
                                               <2 x double> %3)
  ; CHECK-DAG: fmadd.d [[R1]], [[R2]], [[R3]]
  store <2 x double> %4, <2 x double>* %d
  ; CHECK-DAG: st.d [[R1]], 0($4)

  ret void
  ; CHECK: .size fma_v2f64
}

define void @fmsub_v4f32(<4 x float>* %d, <4 x float>* %a, <4 x float>* %b,
                       <4 x float>* %c) nounwind {
  ; CHECK: fmsub_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = load <4 x float>, <4 x float>* %c
  ; CHECK-DAG: ld.w [[R3:\$w[0-9]+]], 0($7)
  %4 = fmul <4 x float> %2, %3
  %5 = fsub <4 x float> %1, %4
  ; CHECK-DAG: fmsub.w [[R1]], [[R2]], [[R3]]
  store <4 x float> %5, <4 x float>* %d
  ; CHECK-DAG: st.w [[R1]], 0($4)

  ret void
  ; CHECK: .size fmsub_v4f32
}

define void @fmsub_v2f64(<2 x double>* %d, <2 x double>* %a, <2 x double>* %b,
                       <2 x double>* %c) nounwind {
  ; CHECK: fmsub_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = load <2 x double>, <2 x double>* %c
  ; CHECK-DAG: ld.d [[R3:\$w[0-9]+]], 0($7)
  %4 = fmul <2 x double> %2, %3
  %5 = fsub <2 x double> %1, %4
  ; CHECK-DAG: fmsub.d [[R1]], [[R2]], [[R3]]
  store <2 x double> %5, <2 x double>* %d
  ; CHECK-DAG: st.d [[R1]], 0($4)

  ret void
  ; CHECK: .size fmsub_v2f64
}

define void @fdiv_v4f32(<4 x float>* %c, <4 x float>* %a, <4 x float>* %b) nounwind {
  ; CHECK: fdiv_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = load <4 x float>, <4 x float>* %b
  ; CHECK-DAG: ld.w [[R2:\$w[0-9]+]], 0($6)
  %3 = fdiv <4 x float> %1, %2
  ; CHECK-DAG: fdiv.w [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size fdiv_v4f32
}

define void @fdiv_v2f64(<2 x double>* %c, <2 x double>* %a, <2 x double>* %b) nounwind {
  ; CHECK: fdiv_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = load <2 x double>, <2 x double>* %b
  ; CHECK-DAG: ld.d [[R2:\$w[0-9]+]], 0($6)
  %3 = fdiv <2 x double> %1, %2
  ; CHECK-DAG: fdiv.d [[R3:\$w[0-9]+]], [[R1]], [[R2]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size fdiv_v2f64
}

define void @fabs_v4f32(<4 x float>* %c, <4 x float>* %a) nounwind {
  ; CHECK: fabs_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <4 x float> @llvm.fabs.v4f32 (<4 x float> %1)
  ; CHECK-DAG: fmax_a.w [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <4 x float> %2, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size fabs_v4f32
}

define void @fabs_v2f64(<2 x double>* %c, <2 x double>* %a) nounwind {
  ; CHECK: fabs_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <2 x double> @llvm.fabs.v2f64 (<2 x double> %1)
  ; CHECK-DAG: fmax_a.d [[R3:\$w[0-9]+]], [[R1]], [[R1]]
  store <2 x double> %2, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size fabs_v2f64
}

define void @fexp2_v4f32(<4 x float>* %c, <4 x float>* %a) nounwind {
  ; CHECK: fexp2_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <4 x float> @llvm.exp2.v4f32 (<4 x float> %1)
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: ffint_u.w [[R4:\$w[0-9]+]], [[R3]]
  ; CHECK-DAG: fexp2.w [[R4:\$w[0-9]+]], [[R3]], [[R1]]
  store <4 x float> %2, <4 x float>* %c
  ; CHECK-DAG: st.w [[R4]], 0($4)

  ret void
  ; CHECK: .size fexp2_v4f32
}

define void @fexp2_v2f64(<2 x double>* %c, <2 x double>* %a) nounwind {
  ; CHECK: fexp2_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <2 x double> @llvm.exp2.v2f64 (<2 x double> %1)
  ; CHECK-DAG: ldi.d [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: ffint_u.d [[R4:\$w[0-9]+]], [[R3]]
  ; CHECK-DAG: fexp2.d [[R4:\$w[0-9]+]], [[R3]], [[R1]]
  store <2 x double> %2, <2 x double>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size fexp2_v2f64
}

define void @fexp2_v4f32_2(<4 x float>* %c, <4 x float>* %a) nounwind {
  ; CHECK: fexp2_v4f32_2:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <4 x float> @llvm.exp2.v4f32 (<4 x float> %1)
  %3 = fmul <4 x float> <float 2.0, float 2.0, float 2.0, float 2.0>, %2
  ; CHECK-DAG: ldi.w [[R3:\$w[0-9]+]], 1
  ; CHECK-DAG: ffint_u.w [[R4:\$w[0-9]+]], [[R3]]
  ; CHECK-DAG: fexp2.w [[R5:\$w[0-9]+]], [[R4]], [[R1]]
  store <4 x float> %3, <4 x float>* %c
  ; CHECK-DAG: st.w [[R5]], 0($4)

  ret void
  ; CHECK: .size fexp2_v4f32_2
}

define void @fexp2_v2f64_2(<2 x double>* %c, <2 x double>* %a) nounwind {
  ; CHECK: fexp2_v2f64_2:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <2 x double> @llvm.exp2.v2f64 (<2 x double> %1)
  %3 = fmul <2 x double> <double 2.0, double 2.0>, %2
  ; CHECK-DAG: ldi.d [[R2:\$w[0-9]+]], 1
  ; CHECK-DAG: ffint_u.d [[R3:\$w[0-9]+]], [[R2]]
  ; CHECK-DAG: fexp2.d [[R4:\$w[0-9]+]], [[R3]], [[R1]]
  store <2 x double> %3, <2 x double>* %c
  ; CHECK-DAG: st.d [[R4]], 0($4)

  ret void
  ; CHECK: .size fexp2_v2f64_2
}

define void @fsqrt_v4f32(<4 x float>* %c, <4 x float>* %a) nounwind {
  ; CHECK: fsqrt_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <4 x float> @llvm.sqrt.v4f32 (<4 x float> %1)
  ; CHECK-DAG: fsqrt.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x float> %2, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size fsqrt_v4f32
}

define void @fsqrt_v2f64(<2 x double>* %c, <2 x double>* %a) nounwind {
  ; CHECK: fsqrt_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = tail call <2 x double> @llvm.sqrt.v2f64 (<2 x double> %1)
  ; CHECK-DAG: fsqrt.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x double> %2, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size fsqrt_v2f64
}

define void @ffint_u_v4f32(<4 x float>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: ffint_u_v4f32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = uitofp <4 x i32> %1 to <4 x float>
  ; CHECK-DAG: ffint_u.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x float> %2, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ffint_u_v4f32
}

define void @ffint_u_v2f64(<2 x double>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: ffint_u_v2f64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = uitofp <2 x i64> %1 to <2 x double>
  ; CHECK-DAG: ffint_u.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x double> %2, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ffint_u_v2f64
}

define void @ffint_s_v4f32(<4 x float>* %c, <4 x i32>* %a) nounwind {
  ; CHECK: ffint_s_v4f32:

  %1 = load <4 x i32>, <4 x i32>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = sitofp <4 x i32> %1 to <4 x float>
  ; CHECK-DAG: ffint_s.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x float> %2, <4 x float>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ffint_s_v4f32
}

define void @ffint_s_v2f64(<2 x double>* %c, <2 x i64>* %a) nounwind {
  ; CHECK: ffint_s_v2f64:

  %1 = load <2 x i64>, <2 x i64>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = sitofp <2 x i64> %1 to <2 x double>
  ; CHECK-DAG: ffint_s.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x double> %2, <2 x double>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ffint_s_v2f64
}

define void @ftrunc_u_v4f32(<4 x i32>* %c, <4 x float>* %a) nounwind {
  ; CHECK: ftrunc_u_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = fptoui <4 x float> %1 to <4 x i32>
  ; CHECK-DAG: ftrunc_u.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ftrunc_u_v4f32
}

define void @ftrunc_u_v2f64(<2 x i64>* %c, <2 x double>* %a) nounwind {
  ; CHECK: ftrunc_u_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = fptoui <2 x double> %1 to <2 x i64>
  ; CHECK-DAG: ftrunc_u.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ftrunc_u_v2f64
}

define void @ftrunc_s_v4f32(<4 x i32>* %c, <4 x float>* %a) nounwind {
  ; CHECK: ftrunc_s_v4f32:

  %1 = load <4 x float>, <4 x float>* %a
  ; CHECK-DAG: ld.w [[R1:\$w[0-9]+]], 0($5)
  %2 = fptosi <4 x float> %1 to <4 x i32>
  ; CHECK-DAG: ftrunc_s.w [[R3:\$w[0-9]+]], [[R1]]
  store <4 x i32> %2, <4 x i32>* %c
  ; CHECK-DAG: st.w [[R3]], 0($4)

  ret void
  ; CHECK: .size ftrunc_s_v4f32
}

define void @ftrunc_s_v2f64(<2 x i64>* %c, <2 x double>* %a) nounwind {
  ; CHECK: ftrunc_s_v2f64:

  %1 = load <2 x double>, <2 x double>* %a
  ; CHECK-DAG: ld.d [[R1:\$w[0-9]+]], 0($5)
  %2 = fptosi <2 x double> %1 to <2 x i64>
  ; CHECK-DAG: ftrunc_s.d [[R3:\$w[0-9]+]], [[R1]]
  store <2 x i64> %2, <2 x i64>* %c
  ; CHECK-DAG: st.d [[R3]], 0($4)

  ret void
  ; CHECK: .size ftrunc_s_v2f64
}

declare <4 x float>  @llvm.fabs.v4f32(<4 x float>  %Val)
declare <2 x double> @llvm.fabs.v2f64(<2 x double> %Val)
declare <4 x float>  @llvm.exp2.v4f32(<4 x float>  %val)
declare <2 x double> @llvm.exp2.v2f64(<2 x double> %val)
declare <4 x float>  @llvm.fma.v4f32(<4 x float>  %a, <4 x float>  %b,
                                     <4 x float>  %c)
declare <2 x double> @llvm.fma.v2f64(<2 x double> %a, <2 x double> %b,
                                     <2 x double> %c)
declare <4 x float>  @llvm.sqrt.v4f32(<4 x float>  %Val)
declare <2 x double> @llvm.sqrt.v2f64(<2 x double> %Val)
