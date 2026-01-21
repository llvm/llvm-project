; RUN: opt < %s -passes=slp-vectorizer -mtriple=riscv64 -mattr=+m,+v -S | FileCheck %s

define void @test_max_reg_vf_boundary(ptr %pl, ptr %ps) {
; CHECK-LABEL: @test_max_reg_vf_boundary(
; ensuring maxregVF slice is vectorized correctly even with the mixed tree sizes
; CHECK:      load <4 x i32>
; CHECK-NEXT: store <4 x i32>

  ; random offsets scalar tests
  %gep_l_unrelated_1 = getelementptr inbounds i32, ptr %pl, i32 100
  %gep_l_unrelated_2 = getelementptr inbounds i32, ptr %pl, i32 200

  ; vf = maxregvf tests
  %gep_l_contiguous = getelementptr inbounds i32, ptr %pl, i32 2
  %gep_l3 = getelementptr inbounds i32, ptr %pl, i32 3
  %gep_l4 = getelementptr inbounds i32, ptr %pl, i32 4
  %gep_l5 = getelementptr inbounds i32, ptr %pl, i32 5

  ; forcing differing tree sizes
  %gep_l_op_mismatch_1 = getelementptr inbounds i32, ptr %pl, i32 300
  %gep_l_op_mismatch_2 = getelementptr inbounds i32, ptr %pl, i32 400

  %load0 = load i32, ptr %gep_l_unrelated_1, align 4
  %load1 = load i32, ptr %gep_l_unrelated_2, align 4
  %load2 = load i32, ptr %gep_l_contiguous, align 4
  %load3 = load i32, ptr %gep_l3, align 4
  %load4 = load i32, ptr %gep_l4, align 4
  %load5 = load i32, ptr %gep_l5, align 4
  %load6 = load i32, ptr %gep_l_op_mismatch_1, align 4
  %load7 = load i32, ptr %gep_l_op_mismatch_2, align 4
  %add6 = add i32 %load6, 1
  %add7 = add i32 %load7, 1

  %gep_s0 = getelementptr inbounds i32, ptr %ps, i32 0
  %gep_s1 = getelementptr inbounds i32, ptr %ps, i32 1
  %gep_s2 = getelementptr inbounds i32, ptr %ps, i32 2
  %gep_s3 = getelementptr inbounds i32, ptr %ps, i32 3
  %gep_s4 = getelementptr inbounds i32, ptr %ps, i32 4
  %gep_s5 = getelementptr inbounds i32, ptr %ps, i32 5
  %gep_s6 = getelementptr inbounds i32, ptr %ps, i32 6
  %gep_s7 = getelementptr inbounds i32, ptr %ps, i32 7

  store i32 %load0, ptr %gep_s0, align 4
  store i32 %load1, ptr %gep_s1, align 4
  store i32 %load2, ptr %gep_s2, align 4
  store i32 %load3, ptr %gep_s3, align 4
  store i32 %load4, ptr %gep_s4, align 4
  store i32 %load5, ptr %gep_s5, align 4
  store i32 %add6, ptr %gep_s6, align 4
  store i32 %add7, ptr %gep_s7, align 4

  ret void
}