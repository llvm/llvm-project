; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Assumed Context:
; CHECK-NEXT: { : }
;
; Make sure the large number of read-only accesses does not cause make us
; invalidate the scop.
;
;    void many_read_only_accesses(float A[], float B[]) {
;      for (long i = 0; i < 1024; i++) {
;        for (long j = 0; j < 1024; j++) {
;          A[j] += B[i] + B[i + 1] + B[i + 2] + B[i + 3] + B[i + 4] + B[i + 5] +
;                  B[i + 6] + B[i + 7] + B[i + 8] + B[i + 9] + B[i + 0] + B[i + 11] +
;                  B[i + 12] + B[i + 13] + B[i + 14] + B[i + 15] + B[i + 16] +
;                  B[i + 17] + B[i + 18] + B[i + 19] + B[i + 10] + B[i + 21] +
;                  B[i + 22] + B[i + 23] + B[i + 24] + B[i + 25] + B[i + 26] +
;                  B[i + 27] + B[i + 28] + B[i + 29] + B[i + 20] + B[i + 31] +
;                  B[i + 32] + B[i + 33] + B[i + 34] + B[i + 35] + B[i + 36] +
;                  B[i + 37] + B[i + 38] + B[i + 39] + B[i + 30];
;        }
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @many_read_only_accesses(float* %A, float* %B) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb172, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp173, %bb172 ]
  %exitcond1 = icmp ne i64 %i.0, 1024
  br i1 %exitcond1, label %bb3, label %bb174

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb169, %bb3
  %j.0 = phi i64 [ 0, %bb3 ], [ %tmp170, %bb169 ]
  %exitcond = icmp ne i64 %j.0, 1024
  br i1 %exitcond, label %bb5, label %bb171

bb5:                                              ; preds = %bb4
  %tmp = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp6 = load float, float* %tmp, align 4
  %tmp7 = add nuw nsw i64 %i.0, 1
  %tmp8 = getelementptr inbounds float, float* %B, i64 %tmp7
  %tmp9 = load float, float* %tmp8, align 4
  %tmp10 = fadd float %tmp6, %tmp9
  %tmp11 = add nuw nsw i64 %i.0, 2
  %tmp12 = getelementptr inbounds float, float* %B, i64 %tmp11
  %tmp13 = load float, float* %tmp12, align 4
  %tmp14 = fadd float %tmp10, %tmp13
  %tmp15 = add nuw nsw i64 %i.0, 3
  %tmp16 = getelementptr inbounds float, float* %B, i64 %tmp15
  %tmp17 = load float, float* %tmp16, align 4
  %tmp18 = fadd float %tmp14, %tmp17
  %tmp19 = add nuw nsw i64 %i.0, 4
  %tmp20 = getelementptr inbounds float, float* %B, i64 %tmp19
  %tmp21 = load float, float* %tmp20, align 4
  %tmp22 = fadd float %tmp18, %tmp21
  %tmp23 = add nuw nsw i64 %i.0, 5
  %tmp24 = getelementptr inbounds float, float* %B, i64 %tmp23
  %tmp25 = load float, float* %tmp24, align 4
  %tmp26 = fadd float %tmp22, %tmp25
  %tmp27 = add nuw nsw i64 %i.0, 6
  %tmp28 = getelementptr inbounds float, float* %B, i64 %tmp27
  %tmp29 = load float, float* %tmp28, align 4
  %tmp30 = fadd float %tmp26, %tmp29
  %tmp31 = add nuw nsw i64 %i.0, 7
  %tmp32 = getelementptr inbounds float, float* %B, i64 %tmp31
  %tmp33 = load float, float* %tmp32, align 4
  %tmp34 = fadd float %tmp30, %tmp33
  %tmp35 = add nuw nsw i64 %i.0, 8
  %tmp36 = getelementptr inbounds float, float* %B, i64 %tmp35
  %tmp37 = load float, float* %tmp36, align 4
  %tmp38 = fadd float %tmp34, %tmp37
  %tmp39 = add nuw nsw i64 %i.0, 9
  %tmp40 = getelementptr inbounds float, float* %B, i64 %tmp39
  %tmp41 = load float, float* %tmp40, align 4
  %tmp42 = fadd float %tmp38, %tmp41
  %tmp43 = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp44 = load float, float* %tmp43, align 4
  %tmp45 = fadd float %tmp42, %tmp44
  %tmp46 = add nuw nsw i64 %i.0, 11
  %tmp47 = getelementptr inbounds float, float* %B, i64 %tmp46
  %tmp48 = load float, float* %tmp47, align 4
  %tmp49 = fadd float %tmp45, %tmp48
  %tmp50 = add nuw nsw i64 %i.0, 12
  %tmp51 = getelementptr inbounds float, float* %B, i64 %tmp50
  %tmp52 = load float, float* %tmp51, align 4
  %tmp53 = fadd float %tmp49, %tmp52
  %tmp54 = add nuw nsw i64 %i.0, 13
  %tmp55 = getelementptr inbounds float, float* %B, i64 %tmp54
  %tmp56 = load float, float* %tmp55, align 4
  %tmp57 = fadd float %tmp53, %tmp56
  %tmp58 = add nuw nsw i64 %i.0, 14
  %tmp59 = getelementptr inbounds float, float* %B, i64 %tmp58
  %tmp60 = load float, float* %tmp59, align 4
  %tmp61 = fadd float %tmp57, %tmp60
  %tmp62 = add nuw nsw i64 %i.0, 15
  %tmp63 = getelementptr inbounds float, float* %B, i64 %tmp62
  %tmp64 = load float, float* %tmp63, align 4
  %tmp65 = fadd float %tmp61, %tmp64
  %tmp66 = add nuw nsw i64 %i.0, 16
  %tmp67 = getelementptr inbounds float, float* %B, i64 %tmp66
  %tmp68 = load float, float* %tmp67, align 4
  %tmp69 = fadd float %tmp65, %tmp68
  %tmp70 = add nuw nsw i64 %i.0, 17
  %tmp71 = getelementptr inbounds float, float* %B, i64 %tmp70
  %tmp72 = load float, float* %tmp71, align 4
  %tmp73 = fadd float %tmp69, %tmp72
  %tmp74 = add nuw nsw i64 %i.0, 18
  %tmp75 = getelementptr inbounds float, float* %B, i64 %tmp74
  %tmp76 = load float, float* %tmp75, align 4
  %tmp77 = fadd float %tmp73, %tmp76
  %tmp78 = add nuw nsw i64 %i.0, 19
  %tmp79 = getelementptr inbounds float, float* %B, i64 %tmp78
  %tmp80 = load float, float* %tmp79, align 4
  %tmp81 = fadd float %tmp77, %tmp80
  %tmp82 = add nuw nsw i64 %i.0, 10
  %tmp83 = getelementptr inbounds float, float* %B, i64 %tmp82
  %tmp84 = load float, float* %tmp83, align 4
  %tmp85 = fadd float %tmp81, %tmp84
  %tmp86 = add nuw nsw i64 %i.0, 21
  %tmp87 = getelementptr inbounds float, float* %B, i64 %tmp86
  %tmp88 = load float, float* %tmp87, align 4
  %tmp89 = fadd float %tmp85, %tmp88
  %tmp90 = add nuw nsw i64 %i.0, 22
  %tmp91 = getelementptr inbounds float, float* %B, i64 %tmp90
  %tmp92 = load float, float* %tmp91, align 4
  %tmp93 = fadd float %tmp89, %tmp92
  %tmp94 = add nuw nsw i64 %i.0, 23
  %tmp95 = getelementptr inbounds float, float* %B, i64 %tmp94
  %tmp96 = load float, float* %tmp95, align 4
  %tmp97 = fadd float %tmp93, %tmp96
  %tmp98 = add nuw nsw i64 %i.0, 24
  %tmp99 = getelementptr inbounds float, float* %B, i64 %tmp98
  %tmp100 = load float, float* %tmp99, align 4
  %tmp101 = fadd float %tmp97, %tmp100
  %tmp102 = add nuw nsw i64 %i.0, 25
  %tmp103 = getelementptr inbounds float, float* %B, i64 %tmp102
  %tmp104 = load float, float* %tmp103, align 4
  %tmp105 = fadd float %tmp101, %tmp104
  %tmp106 = add nuw nsw i64 %i.0, 26
  %tmp107 = getelementptr inbounds float, float* %B, i64 %tmp106
  %tmp108 = load float, float* %tmp107, align 4
  %tmp109 = fadd float %tmp105, %tmp108
  %tmp110 = add nuw nsw i64 %i.0, 27
  %tmp111 = getelementptr inbounds float, float* %B, i64 %tmp110
  %tmp112 = load float, float* %tmp111, align 4
  %tmp113 = fadd float %tmp109, %tmp112
  %tmp114 = add nuw nsw i64 %i.0, 28
  %tmp115 = getelementptr inbounds float, float* %B, i64 %tmp114
  %tmp116 = load float, float* %tmp115, align 4
  %tmp117 = fadd float %tmp113, %tmp116
  %tmp118 = add nuw nsw i64 %i.0, 29
  %tmp119 = getelementptr inbounds float, float* %B, i64 %tmp118
  %tmp120 = load float, float* %tmp119, align 4
  %tmp121 = fadd float %tmp117, %tmp120
  %tmp122 = add nuw nsw i64 %i.0, 20
  %tmp123 = getelementptr inbounds float, float* %B, i64 %tmp122
  %tmp124 = load float, float* %tmp123, align 4
  %tmp125 = fadd float %tmp121, %tmp124
  %tmp126 = add nuw nsw i64 %i.0, 31
  %tmp127 = getelementptr inbounds float, float* %B, i64 %tmp126
  %tmp128 = load float, float* %tmp127, align 4
  %tmp129 = fadd float %tmp125, %tmp128
  %tmp130 = add nuw nsw i64 %i.0, 32
  %tmp131 = getelementptr inbounds float, float* %B, i64 %tmp130
  %tmp132 = load float, float* %tmp131, align 4
  %tmp133 = fadd float %tmp129, %tmp132
  %tmp134 = add nuw nsw i64 %i.0, 33
  %tmp135 = getelementptr inbounds float, float* %B, i64 %tmp134
  %tmp136 = load float, float* %tmp135, align 4
  %tmp137 = fadd float %tmp133, %tmp136
  %tmp138 = add nuw nsw i64 %i.0, 34
  %tmp139 = getelementptr inbounds float, float* %B, i64 %tmp138
  %tmp140 = load float, float* %tmp139, align 4
  %tmp141 = fadd float %tmp137, %tmp140
  %tmp142 = add nuw nsw i64 %i.0, 35
  %tmp143 = getelementptr inbounds float, float* %B, i64 %tmp142
  %tmp144 = load float, float* %tmp143, align 4
  %tmp145 = fadd float %tmp141, %tmp144
  %tmp146 = add nuw nsw i64 %i.0, 36
  %tmp147 = getelementptr inbounds float, float* %B, i64 %tmp146
  %tmp148 = load float, float* %tmp147, align 4
  %tmp149 = fadd float %tmp145, %tmp148
  %tmp150 = add nuw nsw i64 %i.0, 37
  %tmp151 = getelementptr inbounds float, float* %B, i64 %tmp150
  %tmp152 = load float, float* %tmp151, align 4
  %tmp153 = fadd float %tmp149, %tmp152
  %tmp154 = add nuw nsw i64 %i.0, 38
  %tmp155 = getelementptr inbounds float, float* %B, i64 %tmp154
  %tmp156 = load float, float* %tmp155, align 4
  %tmp157 = fadd float %tmp153, %tmp156
  %tmp158 = add nuw nsw i64 %i.0, 39
  %tmp159 = getelementptr inbounds float, float* %B, i64 %tmp158
  %tmp160 = load float, float* %tmp159, align 4
  %tmp161 = fadd float %tmp157, %tmp160
  %tmp162 = add nuw nsw i64 %i.0, 30
  %tmp163 = getelementptr inbounds float, float* %B, i64 %tmp162
  %tmp164 = load float, float* %tmp163, align 4
  %tmp165 = fadd float %tmp161, %tmp164
  %tmp166 = getelementptr inbounds float, float* %A, i64 %j.0
  %tmp167 = load float, float* %tmp166, align 4
  %tmp168 = fadd float %tmp167, %tmp165
  store float %tmp168, float* %tmp166, align 4
  br label %bb169

bb169:                                            ; preds = %bb5
  %tmp170 = add nuw nsw i64 %j.0, 1
  br label %bb4

bb171:                                            ; preds = %bb4
  br label %bb172

bb172:                                            ; preds = %bb171
  %tmp173 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb174:                                            ; preds = %bb2
  ret void
}
