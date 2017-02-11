; RUN: llc -march=hexagon -mcpu=hexagonv60 -enable-bsb-sched=0 -enable-pipeliner < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s | FileCheck %s

; From coremark. Test that we pipeline the matrix multiplication bitextract
; function. The pipelined code should have two packets.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: = extractu([[REG2:(r[0-9]+)]],
; CHECK: = extractu([[REG2]],
; CHECK: [[REG0:(r[0-9]+)]] = memh
; CHECK: [[REG1:(r[0-9]+)]] = memh
; CHECK: += mpyi
; CHECK: [[REG2]] = mpyi([[REG0]],[[REG1]])
; CHECK: endloop0

%union_h2_sem_t = type { i32 }

@sem_i = common global [0 x %union_h2_sem_t] zeroinitializer, align 4

define void @matrix_mul_matrix_bitextract(i32 %N, i32* %C, i16* %A, i16* %B) {
entry:
  %cmp53 = icmp eq i32 %N, 0
  br i1 %cmp53, label %for_end27, label %for_body3_lr_ph_us

for_body3_lr_ph_us:
  %i_054_us = phi i32 [ %inc26_us, %for_cond1_for_inc25_crit_edge_us ], [ 0, %entry ]
  %0 = mul i32 %i_054_us, %N
  %arrayidx9_us_us_gep = getelementptr i16, i16* %A, i32 %0
  br label %for_body3_us_us

for_cond1_for_inc25_crit_edge_us:
  %inc26_us = add i32 %i_054_us, 1
  %exitcond89 = icmp eq i32 %inc26_us, %N
  br i1 %exitcond89, label %for_end27, label %for_body3_lr_ph_us

for_body3_us_us:
  %j_052_us_us = phi i32 [ %inc23_us_us, %for_cond4_for_inc22_crit_edge_us_us ], [ 0, %for_body3_lr_ph_us ]
  %add_us_us = add i32 %j_052_us_us, %0
  %arrayidx_us_us = getelementptr inbounds i32, i32* %C, i32 %add_us_us
  store i32 0, i32* %arrayidx_us_us, align 4
  br label %for_body6_us_us

for_cond4_for_inc22_crit_edge_us_us:
  store i32 %add21_us_us, i32* %arrayidx_us_us, align 4
  %inc23_us_us = add i32 %j_052_us_us, 1
  %exitcond88 = icmp eq i32 %inc23_us_us, %N
  br i1 %exitcond88, label %for_cond1_for_inc25_crit_edge_us, label %for_body3_us_us

for_body6_us_us:
  %1 = phi i32 [ 0, %for_body3_us_us ], [ %add21_us_us, %for_body6_us_us ]
  %arrayidx9_us_us_phi = phi i16* [ %arrayidx9_us_us_gep, %for_body3_us_us ], [ %arrayidx9_us_us_inc, %for_body6_us_us ]
  %k_050_us_us = phi i32 [ 0, %for_body3_us_us ], [ %inc_us_us, %for_body6_us_us ]
  %2 = load i16, i16* %arrayidx9_us_us_phi, align 2
  %conv_us_us = sext i16 %2 to i32
  %mul10_us_us = mul i32 %k_050_us_us, %N
  %add11_us_us = add i32 %mul10_us_us, %j_052_us_us
  %arrayidx12_us_us = getelementptr inbounds i16, i16* %B, i32 %add11_us_us
  %3 = load i16, i16* %arrayidx12_us_us, align 2
  %conv13_us_us = sext i16 %3 to i32
  %mul14_us_us = mul nsw i32 %conv13_us_us, %conv_us_us
  %shr47_us_us = lshr i32 %mul14_us_us, 2
  %and_us_us = and i32 %shr47_us_us, 15
  %shr1548_us_us = lshr i32 %mul14_us_us, 5
  %and16_us_us = and i32 %shr1548_us_us, 127
  %mul17_us_us = mul i32 %and_us_us, %and16_us_us
  %add21_us_us = add i32 %mul17_us_us, %1
  %inc_us_us = add i32 %k_050_us_us, 1
  %exitcond87 = icmp eq i32 %inc_us_us, %N
  %arrayidx9_us_us_inc = getelementptr i16, i16* %arrayidx9_us_us_phi, i32 1
  br i1 %exitcond87, label %for_cond4_for_inc22_crit_edge_us_us, label %for_body6_us_us

for_end27:
  ret void
}
