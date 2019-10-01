; RUN: llc -march=amdgcn < %s -verify-machineinstrs | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s -verify-machineinstrs | FileCheck -check-prefix=SI %s

; If this occurs it is likely due to reordering and the restore was
; originally supposed to happen before SI_END_CF.

; SI: s_or_b64 exec, exec, [[SAVED:s\[[0-9]+:[0-9]+\]|[a-z]+]]
; SI-NOT: v_readlane_b32 [[SAVED]]

define amdgpu_ps void @main() #0 {
main_body:
  %tmp = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 16, i32 0)
  %tmp1 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 32, i32 0)
  %tmp2 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 80, i32 0)
  %tmp3 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 84, i32 0)
  %tmp4 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 88, i32 0)
  %tmp5 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 96, i32 0)
  %tmp6 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 100, i32 0)
  %tmp7 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 104, i32 0)
  %tmp8 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 112, i32 0)
  %tmp9 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 116, i32 0)
  %tmp10 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 120, i32 0)
  %tmp11 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 128, i32 0)
  %tmp12 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 132, i32 0)
  %tmp13 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 136, i32 0)
  %tmp14 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 144, i32 0)
  %tmp15 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 148, i32 0)
  %tmp16 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 152, i32 0)
  %tmp17 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 160, i32 0)
  %tmp18 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 164, i32 0)
  %tmp19 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 168, i32 0)
  %tmp20 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 176, i32 0)
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 180, i32 0)
  %tmp22 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 184, i32 0)
  %tmp23 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 192, i32 0)
  %tmp24 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 196, i32 0)
  %tmp25 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 200, i32 0)
  %tmp26 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 208, i32 0)
  %tmp27 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 212, i32 0)
  %tmp28 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 216, i32 0)
  %tmp29 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 224, i32 0)
  %tmp30 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 228, i32 0)
  %tmp31 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 232, i32 0)
  %tmp32 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 240, i32 0)
  %tmp33 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 244, i32 0)
  %tmp34 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 248, i32 0)
  %tmp35 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 256, i32 0)
  %tmp36 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 260, i32 0)
  %tmp37 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 264, i32 0)
  %tmp38 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 272, i32 0)
  %tmp39 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 276, i32 0)
  %tmp40 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 280, i32 0)
  %tmp41 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 288, i32 0)
  %tmp42 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 292, i32 0)
  %tmp43 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 296, i32 0)
  %tmp44 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 304, i32 0)
  %tmp45 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 308, i32 0)
  %tmp46 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 312, i32 0)
  %tmp47 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 320, i32 0)
  %tmp48 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 324, i32 0)
  %tmp49 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 328, i32 0)
  %tmp50 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 336, i32 0)
  %tmp51 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 340, i32 0)
  %tmp52 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 344, i32 0)
  %tmp53 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 352, i32 0)
  %tmp54 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 356, i32 0)
  %tmp55 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 360, i32 0)
  %tmp56 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 368, i32 0)
  %tmp57 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 372, i32 0)
  %tmp58 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 376, i32 0)
  %tmp59 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 384, i32 0)
  %tmp60 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 388, i32 0)
  %tmp61 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 392, i32 0)
  %tmp62 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 400, i32 0)
  %tmp63 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 404, i32 0)
  %tmp64 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 408, i32 0)
  %tmp65 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 416, i32 0)
  %tmp66 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 420, i32 0)
  br label %LOOP

LOOP:                                             ; preds = %ENDIF2795, %main_body
  %temp894.0 = phi float [ 0.000000e+00, %main_body ], [ %temp894.1, %ENDIF2795 ]
  %temp18.0 = phi float [ undef, %main_body ], [ %temp18.1, %ENDIF2795 ]
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tmp67 = icmp sgt i32 %tid, 4
  br i1 %tmp67, label %ENDLOOP, label %ENDIF

ENDLOOP:                                          ; preds = %ELSE2566, %LOOP
  %one.sub.a.i = fsub float 1.000000e+00, %tmp
  %one.sub.ac.i = fmul float %one.sub.a.i, undef
  %result.i = fadd float fmul (float undef, float undef), %one.sub.ac.i
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float undef, float %result.i, float undef, float 1.000000e+00, i1 true, i1 true) #0
  ret void

ENDIF:                                            ; preds = %LOOP
  %tmp68 = fsub float %tmp2, undef
  %tmp69 = fsub float %tmp3, undef
  %tmp70 = fsub float %tmp4, undef
  %tmp71 = fmul float %tmp68, 0.000000e+00
  %tmp72 = fmul float %tmp69, undef
  %tmp73 = fmul float %tmp70, undef
  %tmp74 = fsub float %tmp6, undef
  %tmp75 = fsub float %tmp7, undef
  %tmp76 = fmul float %tmp74, undef
  %tmp77 = fmul float %tmp75, 0.000000e+00
  %tmp78 = call float @llvm.minnum.f32(float %tmp73, float %tmp77)
  %tmp79 = call float @llvm.maxnum.f32(float %tmp71, float 0.000000e+00)
  %tmp80 = call float @llvm.maxnum.f32(float %tmp72, float %tmp76)
  %tmp81 = call float @llvm.maxnum.f32(float undef, float %tmp78)
  %tmp82 = call float @llvm.minnum.f32(float %tmp79, float %tmp80)
  %tmp83 = call float @llvm.minnum.f32(float %tmp82, float undef)
  %tmp84 = fsub float %tmp14, undef
  %tmp85 = fsub float %tmp15, undef
  %tmp86 = fsub float %tmp16, undef
  %tmp87 = fmul float %tmp84, undef
  %tmp88 = fmul float %tmp85, undef
  %tmp89 = fmul float %tmp86, undef
  %tmp90 = fsub float %tmp17, undef
  %tmp91 = fsub float %tmp18, undef
  %tmp92 = fsub float %tmp19, undef
  %tmp93 = fmul float %tmp90, 0.000000e+00
  %tmp94 = fmul float %tmp91, undef
  %tmp95 = fmul float %tmp92, undef
  %tmp96 = call float @llvm.minnum.f32(float %tmp88, float %tmp94)
  %tmp97 = call float @llvm.maxnum.f32(float %tmp87, float %tmp93)
  %tmp98 = call float @llvm.maxnum.f32(float %tmp89, float %tmp95)
  %tmp99 = call float @llvm.maxnum.f32(float undef, float %tmp96)
  %tmp100 = call float @llvm.maxnum.f32(float %tmp99, float undef)
  %tmp101 = call float @llvm.minnum.f32(float %tmp97, float undef)
  %tmp102 = call float @llvm.minnum.f32(float %tmp101, float %tmp98)
  %tmp103 = fsub float %tmp30, undef
  %tmp104 = fsub float %tmp31, undef
  %tmp105 = fmul float %tmp103, 0.000000e+00
  %tmp106 = fmul float %tmp104, 0.000000e+00
  %tmp107 = call float @llvm.minnum.f32(float undef, float %tmp105)
  %tmp108 = call float @llvm.maxnum.f32(float undef, float %tmp106)
  %tmp109 = call float @llvm.maxnum.f32(float undef, float %tmp107)
  %tmp110 = call float @llvm.maxnum.f32(float %tmp109, float undef)
  %tmp111 = call float @llvm.minnum.f32(float undef, float %tmp108)
  %tmp112 = fsub float %tmp32, undef
  %tmp113 = fsub float %tmp33, undef
  %tmp114 = fsub float %tmp34, undef
  %tmp115 = fmul float %tmp112, 0.000000e+00
  %tmp116 = fmul float %tmp113, undef
  %tmp117 = fmul float %tmp114, undef
  %tmp118 = fsub float %tmp35, undef
  %tmp119 = fsub float %tmp36, undef
  %tmp120 = fsub float %tmp37, undef
  %tmp121 = fmul float %tmp118, undef
  %tmp122 = fmul float %tmp119, undef
  %tmp123 = fmul float %tmp120, undef
  %tmp124 = call float @llvm.minnum.f32(float %tmp115, float %tmp121)
  %tmp125 = call float @llvm.minnum.f32(float %tmp116, float %tmp122)
  %tmp126 = call float @llvm.minnum.f32(float %tmp117, float %tmp123)
  %tmp127 = call float @llvm.maxnum.f32(float %tmp124, float %tmp125)
  %tmp128 = call float @llvm.maxnum.f32(float %tmp127, float %tmp126)
  %tmp129 = fsub float %tmp38, undef
  %tmp130 = fsub float %tmp39, undef
  %tmp131 = fsub float %tmp40, undef
  %tmp132 = fmul float %tmp129, 0.000000e+00
  %tmp133 = fmul float %tmp130, undef
  %tmp134 = fmul float %tmp131, undef
  %tmp135 = fsub float %tmp41, undef
  %tmp136 = fsub float %tmp42, undef
  %tmp137 = fsub float %tmp43, undef
  %tmp138 = fmul float %tmp135, undef
  %tmp139 = fmul float %tmp136, undef
  %tmp140 = fmul float %tmp137, undef
  %tmp141 = call float @llvm.minnum.f32(float %tmp132, float %tmp138)
  %tmp142 = call float @llvm.minnum.f32(float %tmp133, float %tmp139)
  %tmp143 = call float @llvm.minnum.f32(float %tmp134, float %tmp140)
  %tmp144 = call float @llvm.maxnum.f32(float %tmp141, float %tmp142)
  %tmp145 = call float @llvm.maxnum.f32(float %tmp144, float %tmp143)
  %tmp146 = fsub float %tmp44, undef
  %tmp147 = fsub float %tmp45, undef
  %tmp148 = fsub float %tmp46, undef
  %tmp149 = fmul float %tmp146, 0.000000e+00
  %tmp150 = fmul float %tmp147, 0.000000e+00
  %tmp151 = fmul float %tmp148, undef
  %tmp152 = fsub float %tmp47, undef
  %tmp153 = fsub float %tmp48, undef
  %tmp154 = fsub float %tmp49, undef
  %tmp155 = fmul float %tmp152, undef
  %tmp156 = fmul float %tmp153, 0.000000e+00
  %tmp157 = fmul float %tmp154, undef
  %tmp158 = call float @llvm.minnum.f32(float %tmp149, float %tmp155)
  %tmp159 = call float @llvm.minnum.f32(float %tmp150, float %tmp156)
  %tmp160 = call float @llvm.minnum.f32(float %tmp151, float %tmp157)
  %tmp161 = call float @llvm.maxnum.f32(float %tmp158, float %tmp159)
  %tmp162 = call float @llvm.maxnum.f32(float %tmp161, float %tmp160)
  %tmp163 = fsub float %tmp50, undef
  %tmp164 = fsub float %tmp51, undef
  %tmp165 = fsub float %tmp52, undef
  %tmp166 = fmul float %tmp163, undef
  %tmp167 = fmul float %tmp164, 0.000000e+00
  %tmp168 = fmul float %tmp165, 0.000000e+00
  %tmp169 = fsub float %tmp53, undef
  %tmp170 = fsub float %tmp54, undef
  %tmp171 = fsub float %tmp55, undef
  %tmp172 = fdiv float 1.000000e+00, %temp18.0
  %tmp173 = fmul float %tmp169, undef
  %tmp174 = fmul float %tmp170, undef
  %tmp175 = fmul float %tmp171, %tmp172
  %tmp176 = call float @llvm.minnum.f32(float %tmp166, float %tmp173)
  %tmp177 = call float @llvm.minnum.f32(float %tmp167, float %tmp174)
  %tmp178 = call float @llvm.minnum.f32(float %tmp168, float %tmp175)
  %tmp179 = call float @llvm.maxnum.f32(float %tmp176, float %tmp177)
  %tmp180 = call float @llvm.maxnum.f32(float %tmp179, float %tmp178)
  %tmp181 = fsub float %tmp62, undef
  %tmp182 = fsub float %tmp63, undef
  %tmp183 = fsub float %tmp64, undef
  %tmp184 = fmul float %tmp181, 0.000000e+00
  %tmp185 = fmul float %tmp182, undef
  %tmp186 = fmul float %tmp183, undef
  %tmp187 = fsub float %tmp65, undef
  %tmp188 = fsub float %tmp66, undef
  %tmp189 = fmul float %tmp187, undef
  %tmp190 = fmul float %tmp188, undef
  %tmp191 = call float @llvm.maxnum.f32(float %tmp184, float %tmp189)
  %tmp192 = call float @llvm.maxnum.f32(float %tmp185, float %tmp190)
  %tmp193 = call float @llvm.maxnum.f32(float %tmp186, float undef)
  %tmp194 = call float @llvm.minnum.f32(float %tmp191, float %tmp192)
  %tmp195 = call float @llvm.minnum.f32(float %tmp194, float %tmp193)
  %.temp292.7 = select i1 undef, float %tmp162, float undef
  %temp292.9 = select i1 false, float %tmp180, float %.temp292.7
  %.temp292.9 = select i1 undef, float undef, float %temp292.9
  %tmp196 = fcmp ogt float undef, 0.000000e+00
  %tmp197 = fcmp olt float undef, %tmp195
  %tmp198 = and i1 %tmp196, %tmp197
  %tmp199 = fcmp olt float undef, %.temp292.9
  %tmp200 = and i1 %tmp198, %tmp199
  %temp292.11 = select i1 %tmp200, float undef, float %.temp292.9
  %tid0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %cmp0 = icmp eq i32 %tid0, 0
  br i1 %cmp0, label %IF2565, label %ELSE2566

IF2565:                                           ; preds = %ENDIF
  %tid1 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %cmp1 = icmp eq i32 %tid1, 0
  br i1 %cmp1, label %ENDIF2582, label %ELSE2584

ELSE2566:                                         ; preds = %ENDIF
  %tid2 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tidf = bitcast i32 %tid2 to float
  %tmp201 = fcmp oeq float %temp292.11, %tidf
  br i1 %tmp201, label %ENDLOOP, label %ELSE2593

ENDIF2564:                                        ; preds = %ENDIF2594, %ENDIF2588
  %temp894.1 = phi float [ undef, %ENDIF2588 ], [ %temp894.2, %ENDIF2594 ]
  %temp18.1 = phi float [ %tmp218, %ENDIF2588 ], [ undef, %ENDIF2594 ]
  %tmp202 = fsub float %tmp5, undef
  %tmp203 = fmul float %tmp202, undef
  %tmp204 = call float @llvm.maxnum.f32(float undef, float %tmp203)
  %tmp205 = call float @llvm.minnum.f32(float %tmp204, float undef)
  %tmp206 = call float @llvm.minnum.f32(float %tmp205, float undef)
  %tmp207 = fcmp ogt float undef, 0.000000e+00
  %tmp208 = fcmp olt float undef, 1.000000e+00
  %tmp209 = and i1 %tmp207, %tmp208
  %tid3 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tidf3 = bitcast i32 %tid3 to float
  %tmp210 = fcmp olt float %tidf3, %tmp206
  %tmp211 = and i1 %tmp209, %tmp210
  br i1 %tmp211, label %ENDIF2795, label %ELSE2797

ELSE2584:                                         ; preds = %IF2565
  br label %ENDIF2582

ENDIF2582:                                        ; preds = %ELSE2584, %IF2565
  %tmp212 = fadd float %tmp1, undef
  %tmp213 = fadd float 0.000000e+00, %tmp212
  %floor = call float @llvm.floor.f32(float %tmp213)
  %tmp214 = fsub float %tmp213, %floor
  %tid4 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %cmp4 = icmp eq i32 %tid4, 0
  br i1 %cmp4, label %IF2589, label %ELSE2590

IF2589:                                           ; preds = %ENDIF2582
  br label %ENDIF2588

ELSE2590:                                         ; preds = %ENDIF2582
  br label %ENDIF2588

ENDIF2588:                                        ; preds = %ELSE2590, %IF2589
  %tmp215 = fsub float 1.000000e+00, %tmp214
  %tmp216 = call float @llvm.sqrt.f32(float %tmp215)
  %tmp217 = fmul float %tmp216, undef
  %tmp218 = fadd float %tmp217, undef
  br label %ENDIF2564

ELSE2593:                                         ; preds = %ELSE2566
  %tmp219 = fcmp oeq float %temp292.11, %tmp81
  %tmp220 = fcmp olt float %tmp81, %tmp83
  %tmp221 = and i1 %tmp219, %tmp220
  br i1 %tmp221, label %ENDIF2594, label %ELSE2596

ELSE2596:                                         ; preds = %ELSE2593
  %tmp222 = fcmp oeq float %temp292.11, %tmp100
  %tmp223 = fcmp olt float %tmp100, %tmp102
  %tmp224 = and i1 %tmp222, %tmp223
  br i1 %tmp224, label %ENDIF2594, label %ELSE2632

ENDIF2594:                                        ; preds = %ELSE2788, %ELSE2785, %ELSE2782, %ELSE2779, %IF2775, %ELSE2761, %ELSE2758, %IF2757, %ELSE2704, %ELSE2686, %ELSE2671, %ELSE2668, %IF2667, %ELSE2632, %ELSE2596, %ELSE2593
  %temp894.2 = phi float [ 0.000000e+00, %IF2667 ], [ 0.000000e+00, %ELSE2671 ], [ 0.000000e+00, %IF2757 ], [ 0.000000e+00, %ELSE2761 ], [ %temp894.0, %ELSE2758 ], [ 0.000000e+00, %IF2775 ], [ 0.000000e+00, %ELSE2779 ], [ 0.000000e+00, %ELSE2782 ], [ %.2848, %ELSE2788 ], [ 0.000000e+00, %ELSE2785 ], [ 0.000000e+00, %ELSE2593 ], [ 0.000000e+00, %ELSE2632 ], [ 0.000000e+00, %ELSE2704 ], [ 0.000000e+00, %ELSE2686 ], [ 0.000000e+00, %ELSE2668 ], [ 0.000000e+00, %ELSE2596 ]
  %tmp225 = fmul float %temp894.2, undef
  br label %ENDIF2564

ELSE2632:                                         ; preds = %ELSE2596
  br i1 undef, label %ENDIF2594, label %ELSE2650

ELSE2650:                                         ; preds = %ELSE2632
  %tmp226 = fcmp oeq float %temp292.11, %tmp110
  %tmp227 = fcmp olt float %tmp110, %tmp111
  %tmp228 = and i1 %tmp226, %tmp227
  br i1 %tmp228, label %IF2667, label %ELSE2668

IF2667:                                           ; preds = %ELSE2650
  br i1 undef, label %ENDIF2594, label %ELSE2671

ELSE2668:                                         ; preds = %ELSE2650
  %tmp229 = fcmp oeq float %temp292.11, %tmp128
  %tmp230 = fcmp olt float %tmp128, undef
  %tmp231 = and i1 %tmp229, %tmp230
  br i1 %tmp231, label %ENDIF2594, label %ELSE2686

ELSE2671:                                         ; preds = %IF2667
  br label %ENDIF2594

ELSE2686:                                         ; preds = %ELSE2668
  %tmp232 = fcmp oeq float %temp292.11, %tmp145
  %tmp233 = fcmp olt float %tmp145, undef
  %tmp234 = and i1 %tmp232, %tmp233
  br i1 %tmp234, label %ENDIF2594, label %ELSE2704

ELSE2704:                                         ; preds = %ELSE2686
  %tmp235 = fcmp oeq float %temp292.11, %tmp180
  %tmp236 = fcmp olt float %tmp180, undef
  %tmp237 = and i1 %tmp235, %tmp236
  br i1 %tmp237, label %ENDIF2594, label %ELSE2740

ELSE2740:                                         ; preds = %ELSE2704
  br i1 undef, label %IF2757, label %ELSE2758

IF2757:                                           ; preds = %ELSE2740
  br i1 undef, label %ENDIF2594, label %ELSE2761

ELSE2758:                                         ; preds = %ELSE2740
  br i1 undef, label %IF2775, label %ENDIF2594

ELSE2761:                                         ; preds = %IF2757
  br label %ENDIF2594

IF2775:                                           ; preds = %ELSE2758
  %tmp238 = fcmp olt float undef, undef
  br i1 %tmp238, label %ENDIF2594, label %ELSE2779

ELSE2779:                                         ; preds = %IF2775
  br i1 undef, label %ENDIF2594, label %ELSE2782

ELSE2782:                                         ; preds = %ELSE2779
  br i1 undef, label %ENDIF2594, label %ELSE2785

ELSE2785:                                         ; preds = %ELSE2782
  %tmp239 = fcmp olt float undef, 0.000000e+00
  br i1 %tmp239, label %ENDIF2594, label %ELSE2788

ELSE2788:                                         ; preds = %ELSE2785
  %tmp240 = fcmp olt float 0.000000e+00, undef
  %.2848 = select i1 %tmp240, float -1.000000e+00, float 1.000000e+00
  br label %ENDIF2594

ELSE2797:                                         ; preds = %ENDIF2564
  %tmp241 = fsub float %tmp8, undef
  %tmp242 = fsub float %tmp9, undef
  %tmp243 = fsub float %tmp10, undef
  %tmp244 = fmul float %tmp241, undef
  %tmp245 = fmul float %tmp242, undef
  %tmp246 = fmul float %tmp243, undef
  %tmp247 = fsub float %tmp11, undef
  %tmp248 = fsub float %tmp12, undef
  %tmp249 = fsub float %tmp13, undef
  %tmp250 = fmul float %tmp247, undef
  %tmp251 = fmul float %tmp248, undef
  %tmp252 = fmul float %tmp249, undef
  %tmp253 = call float @llvm.minnum.f32(float %tmp244, float %tmp250)
  %tmp254 = call float @llvm.minnum.f32(float %tmp245, float %tmp251)
  %tmp255 = call float @llvm.maxnum.f32(float %tmp246, float %tmp252)
  %tmp256 = call float @llvm.maxnum.f32(float %tmp253, float %tmp254)
  %tmp257 = call float @llvm.maxnum.f32(float %tmp256, float undef)
  %tmp258 = call float @llvm.minnum.f32(float undef, float %tmp255)
  %tmp259 = fcmp ogt float %tmp257, 0.000000e+00
  %tmp260 = fcmp olt float %tmp257, 1.000000e+00
  %tmp261 = and i1 %tmp259, %tmp260
  %tmp262 = fcmp olt float %tmp257, %tmp258
  %tmp263 = and i1 %tmp261, %tmp262
  br i1 %tmp263, label %ENDIF2795, label %ELSE2800

ENDIF2795:                                        ; preds = %ELSE2824, %ELSE2821, %ELSE2818, %ELSE2815, %ELSE2812, %ELSE2809, %ELSE2806, %ELSE2803, %ELSE2800, %ELSE2797, %ENDIF2564
  br label %LOOP

ELSE2800:                                         ; preds = %ELSE2797
  br i1 undef, label %ENDIF2795, label %ELSE2803

ELSE2803:                                         ; preds = %ELSE2800
  %tmp264 = fsub float %tmp20, undef
  %tmp265 = fsub float %tmp21, undef
  %tmp266 = fsub float %tmp22, undef
  %tmp267 = fmul float %tmp264, undef
  %tmp268 = fmul float %tmp265, undef
  %tmp269 = fmul float %tmp266, 0.000000e+00
  %tmp270 = fsub float %tmp23, undef
  %tmp271 = fsub float %tmp24, undef
  %tmp272 = fsub float %tmp25, undef
  %tmp273 = fmul float %tmp270, undef
  %tmp274 = fmul float %tmp271, undef
  %tmp275 = fmul float %tmp272, undef
  %tmp276 = call float @llvm.minnum.f32(float %tmp267, float %tmp273)
  %tmp277 = call float @llvm.maxnum.f32(float %tmp268, float %tmp274)
  %tmp278 = call float @llvm.maxnum.f32(float %tmp269, float %tmp275)
  %tmp279 = call float @llvm.maxnum.f32(float %tmp276, float undef)
  %tmp280 = call float @llvm.maxnum.f32(float %tmp279, float undef)
  %tmp281 = call float @llvm.minnum.f32(float undef, float %tmp277)
  %tmp282 = call float @llvm.minnum.f32(float %tmp281, float %tmp278)
  %tmp283 = fcmp ogt float %tmp280, 0.000000e+00
  %tmp284 = fcmp olt float %tmp280, 1.000000e+00
  %tmp285 = and i1 %tmp283, %tmp284
  %tmp286 = fcmp olt float %tmp280, %tmp282
  %tmp287 = and i1 %tmp285, %tmp286
  br i1 %tmp287, label %ENDIF2795, label %ELSE2806

ELSE2806:                                         ; preds = %ELSE2803
  %tmp288 = fsub float %tmp26, undef
  %tmp289 = fsub float %tmp27, undef
  %tmp290 = fsub float %tmp28, undef
  %tmp291 = fmul float %tmp288, undef
  %tmp292 = fmul float %tmp289, 0.000000e+00
  %tmp293 = fmul float %tmp290, undef
  %tmp294 = fsub float %tmp29, undef
  %tmp295 = fmul float %tmp294, undef
  %tmp296 = call float @llvm.minnum.f32(float %tmp291, float %tmp295)
  %tmp297 = call float @llvm.minnum.f32(float %tmp292, float undef)
  %tmp298 = call float @llvm.maxnum.f32(float %tmp293, float undef)
  %tmp299 = call float @llvm.maxnum.f32(float %tmp296, float %tmp297)
  %tmp300 = call float @llvm.maxnum.f32(float %tmp299, float undef)
  %tmp301 = call float @llvm.minnum.f32(float undef, float %tmp298)
  %tmp302 = fcmp ogt float %tmp300, 0.000000e+00
  %tmp303 = fcmp olt float %tmp300, 1.000000e+00
  %tmp304 = and i1 %tmp302, %tmp303
  %tmp305 = fcmp olt float %tmp300, %tmp301
  %tmp306 = and i1 %tmp304, %tmp305
  br i1 %tmp306, label %ENDIF2795, label %ELSE2809

ELSE2809:                                         ; preds = %ELSE2806
  br i1 undef, label %ENDIF2795, label %ELSE2812

ELSE2812:                                         ; preds = %ELSE2809
  br i1 undef, label %ENDIF2795, label %ELSE2815

ELSE2815:                                         ; preds = %ELSE2812
  br i1 undef, label %ENDIF2795, label %ELSE2818

ELSE2818:                                         ; preds = %ELSE2815
  br i1 undef, label %ENDIF2795, label %ELSE2821

ELSE2821:                                         ; preds = %ELSE2818
  %tmp307 = fsub float %tmp56, undef
  %tmp308 = fsub float %tmp57, undef
  %tmp309 = fsub float %tmp58, undef
  %tmp310 = fmul float %tmp307, undef
  %tmp311 = fmul float %tmp308, 0.000000e+00
  %tmp312 = fmul float %tmp309, undef
  %tmp313 = fsub float %tmp59, undef
  %tmp314 = fsub float %tmp60, undef
  %tmp315 = fsub float %tmp61, undef
  %tmp316 = fmul float %tmp313, undef
  %tmp317 = fmul float %tmp314, undef
  %tmp318 = fmul float %tmp315, undef
  %tmp319 = call float @llvm.maxnum.f32(float %tmp310, float %tmp316)
  %tmp320 = call float @llvm.maxnum.f32(float %tmp311, float %tmp317)
  %tmp321 = call float @llvm.maxnum.f32(float %tmp312, float %tmp318)
  %tmp322 = call float @llvm.minnum.f32(float %tmp319, float %tmp320)
  %tmp323 = call float @llvm.minnum.f32(float %tmp322, float %tmp321)
  %tmp324 = fcmp ogt float undef, 0.000000e+00
  %tmp325 = fcmp olt float undef, 1.000000e+00
  %tmp326 = and i1 %tmp324, %tmp325
  %tmp327 = fcmp olt float undef, %tmp323
  %tmp328 = and i1 %tmp326, %tmp327
  br i1 %tmp328, label %ENDIF2795, label %ELSE2824

ELSE2824:                                         ; preds = %ELSE2821
  %.2849 = select i1 undef, float 0.000000e+00, float 1.000000e+00
  br label %ENDIF2795
}

declare float @llvm.floor.f32(float) #1
declare float @llvm.sqrt.f32(float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare float @llvm.amdgcn.s.buffer.load.f32(<4 x i32>, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
