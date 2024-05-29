; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}load_fma_store
; GCN-NOT: scratch_store
; ScratchSize: 0{{$}}

define amdgpu_kernel void @load_fma_store(ptr addrspace(3) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i1 %cc) {
bb:
  %vgpr0 = call <32 x i32> asm sideeffect "; def $0","=${v[0:31]}"()
  %vgpr1 = call <32 x i32> asm sideeffect "; def $0","=${v[32:63]}"()
  %vgpr2 = call <32 x i32> asm sideeffect "; def $0","=${v[64:95]}"()
  %vgpr3 = call <32 x i32> asm sideeffect "; def $0","=${v[96:127]}"()
  br i1 %cc, label %bb1, label %bb2

bb1:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 1
  %tmp2 = load float, ptr addrspace(3) %tmp, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp5 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 3
  %tmp6 = load float, ptr addrspace(3) %tmp5, align 4
  %tmp7 = tail call float @llvm.fmuladd.f32(float %tmp2, float %tmp4, float %tmp6)
  %tmp8 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 5
  %tmp9 = load float, ptr addrspace(3) %tmp8, align 4
  %tmp10 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 6
  %tmp11 = load float, ptr addrspace(3) %tmp10, align 4
  %tmp12 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 7
  %tmp13 = load float, ptr addrspace(3) %tmp12, align 4
  %tmp14 = tail call float @llvm.fmuladd.f32(float %tmp9, float %tmp11, float %tmp13)
  %tmp15 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 9
  %tmp16 = load float, ptr addrspace(3) %tmp15, align 4
  %tmp17 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 10
  %tmp18 = load float, ptr addrspace(3) %tmp17, align 4
  %tmp19 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 11
  %tmp20 = load float, ptr addrspace(3) %tmp19, align 4
  %tmp21 = tail call float @llvm.fmuladd.f32(float %tmp16, float %tmp18, float %tmp20)
  %tmp22 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 13
  %tmp23 = load float, ptr addrspace(3) %tmp22, align 4
  %tmp24 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 14
  %tmp25 = load float, ptr addrspace(3) %tmp24, align 4
  %tmp26 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 15
  %tmp27 = load float, ptr addrspace(3) %tmp26, align 4
  %tmp28 = tail call float @llvm.fmuladd.f32(float %tmp23, float %tmp25, float %tmp27)
  %tmp29 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 17
  %tmp30 = load float, ptr addrspace(3) %tmp29, align 4
  %tmp31 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 18
  %tmp32 = load float, ptr addrspace(3) %tmp31, align 4
  %tmp33 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 19
  %tmp34 = load float, ptr addrspace(3) %tmp33, align 4
  %tmp35 = tail call float @llvm.fmuladd.f32(float %tmp30, float %tmp32, float %tmp34)
  %tmp36 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 21
  %tmp37 = load float, ptr addrspace(3) %tmp36, align 4
  %tmp38 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 22
  %tmp39 = load float, ptr addrspace(3) %tmp38, align 4
  %tmp40 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 23
  %tmp41 = load float, ptr addrspace(3) %tmp40, align 4
  %tmp42 = tail call float @llvm.fmuladd.f32(float %tmp37, float %tmp39, float %tmp41)
  %tmp43 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 25
  %tmp44 = load float, ptr addrspace(3) %tmp43, align 4
  %tmp45 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 26
  %tmp46 = load float, ptr addrspace(3) %tmp45, align 4
  %tmp47 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 27
  %tmp48 = load float, ptr addrspace(3) %tmp47, align 4
  %tmp49 = tail call float @llvm.fmuladd.f32(float %tmp44, float %tmp46, float %tmp48)
  %tmp50 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 29
  %tmp51 = load float, ptr addrspace(3) %tmp50, align 4
  %tmp52 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 30
  %tmp53 = load float, ptr addrspace(3) %tmp52, align 4
  %tmp54 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 31
  %tmp55 = load float, ptr addrspace(3) %tmp54, align 4
  %tmp56 = tail call float @llvm.fmuladd.f32(float %tmp51, float %tmp53, float %tmp55)
  %tmp57 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 33
  %tmp58 = load float, ptr addrspace(3) %tmp57, align 4
  %tmp59 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 34
  %tmp60 = load float, ptr addrspace(3) %tmp59, align 4
  %tmp61 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 35
  %tmp62 = load float, ptr addrspace(3) %tmp61, align 4
  %tmp63 = tail call float @llvm.fmuladd.f32(float %tmp58, float %tmp60, float %tmp62)
  %tmp64 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 37
  %tmp65 = load float, ptr addrspace(3) %tmp64, align 4
  %tmp66 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 38
  %tmp67 = load float, ptr addrspace(3) %tmp66, align 4
  %tmp68 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 39
  %tmp69 = load float, ptr addrspace(3) %tmp68, align 4
  %tmp70 = tail call float @llvm.fmuladd.f32(float %tmp65, float %tmp67, float %tmp69)
  %tmp71 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 41
  %tmp72 = load float, ptr addrspace(3) %tmp71, align 4
  %tmp73 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 42
  %tmp74 = load float, ptr addrspace(3) %tmp73, align 4
  %tmp75 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 43
  %tmp76 = load float, ptr addrspace(3) %tmp75, align 4
  %tmp77 = tail call float @llvm.fmuladd.f32(float %tmp72, float %tmp74, float %tmp76)
  %tmp78 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 45
  %tmp79 = load float, ptr addrspace(3) %tmp78, align 4
  %tmp80 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 46
  %tmp81 = load float, ptr addrspace(3) %tmp80, align 4
  %tmp82 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 47
  %tmp83 = load float, ptr addrspace(3) %tmp82, align 4
  %tmp84 = tail call float @llvm.fmuladd.f32(float %tmp79, float %tmp81, float %tmp83)
  %tmp85 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 49
  %tmp86 = load float, ptr addrspace(3) %tmp85, align 4
  %tmp87 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 50
  %tmp88 = load float, ptr addrspace(3) %tmp87, align 4
  %tmp89 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 51
  %tmp90 = load float, ptr addrspace(3) %tmp89, align 4
  %tmp91 = tail call float @llvm.fmuladd.f32(float %tmp86, float %tmp88, float %tmp90)
  %tmp92 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 53
  %tmp93 = load float, ptr addrspace(3) %tmp92, align 4
  %tmp94 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 54
  %tmp95 = load float, ptr addrspace(3) %tmp94, align 4
  %tmp96 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 55
  %tmp97 = load float, ptr addrspace(3) %tmp96, align 4
  %tmp98 = tail call float @llvm.fmuladd.f32(float %tmp93, float %tmp95, float %tmp97)
  %tmp99 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 57
  %tmp100 = load float, ptr addrspace(3) %tmp99, align 4
  %tmp101 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 58
  %tmp102 = load float, ptr addrspace(3) %tmp101, align 4
  %tmp103 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 59
  %tmp104 = load float, ptr addrspace(3) %tmp103, align 4
  %tmp105 = tail call float @llvm.fmuladd.f32(float %tmp100, float %tmp102, float %tmp104)
  %tmp106 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 61
  %tmp107 = load float, ptr addrspace(3) %tmp106, align 4
  %tmp108 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 62
  %tmp109 = load float, ptr addrspace(3) %tmp108, align 4
  %tmp110 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 63
  %tmp111 = load float, ptr addrspace(3) %tmp110, align 4
  %tmp112 = tail call float @llvm.fmuladd.f32(float %tmp107, float %tmp109, float %tmp111)
  %tmp113 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 65
  %tmp114 = load float, ptr addrspace(3) %tmp113, align 4
  %tmp115 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 66
  %tmp116 = load float, ptr addrspace(3) %tmp115, align 4
  %tmp117 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 67
  %tmp118 = load float, ptr addrspace(3) %tmp117, align 4
  %tmp119 = tail call float @llvm.fmuladd.f32(float %tmp114, float %tmp116, float %tmp118)
  %tmp120 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 69
  %tmp121 = load float, ptr addrspace(3) %tmp120, align 4
  %tmp122 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 70
  %tmp123 = load float, ptr addrspace(3) %tmp122, align 4
  %tmp124 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 71
  %tmp125 = load float, ptr addrspace(3) %tmp124, align 4
  %tmp126 = tail call float @llvm.fmuladd.f32(float %tmp121, float %tmp123, float %tmp125)
  %tmp127 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 73
  %tmp128 = load float, ptr addrspace(3) %tmp127, align 4
  %tmp129 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 74
  %tmp130 = load float, ptr addrspace(3) %tmp129, align 4
  %tmp131 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 75
  %tmp132 = load float, ptr addrspace(3) %tmp131, align 4
  %tmp133 = tail call float @llvm.fmuladd.f32(float %tmp128, float %tmp130, float %tmp132)
  %tmp134 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 77
  %tmp135 = load float, ptr addrspace(3) %tmp134, align 4
  %tmp136 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 78
  %tmp137 = load float, ptr addrspace(3) %tmp136, align 4
  %tmp138 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 79
  %tmp139 = load float, ptr addrspace(3) %tmp138, align 4
  %tmp140 = tail call float @llvm.fmuladd.f32(float %tmp135, float %tmp137, float %tmp139)
  %tmp141 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 81
  %tmp142 = load float, ptr addrspace(3) %tmp141, align 4
  %tmp143 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 82
  %tmp144 = load float, ptr addrspace(3) %tmp143, align 4
  %tmp145 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 83
  %tmp146 = load float, ptr addrspace(3) %tmp145, align 4
  %tmp147 = tail call float @llvm.fmuladd.f32(float %tmp142, float %tmp144, float %tmp146)
  %tmp148 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 85
  %tmp149 = load float, ptr addrspace(3) %tmp148, align 4
  %tmp150 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 86
  %tmp151 = load float, ptr addrspace(3) %tmp150, align 4
  %tmp152 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 87
  %tmp153 = load float, ptr addrspace(3) %tmp152, align 4
  %tmp154 = tail call float @llvm.fmuladd.f32(float %tmp149, float %tmp151, float %tmp153)
  %tmp155 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 89
  %tmp156 = load float, ptr addrspace(3) %tmp155, align 4
  %tmp157 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 90
  %tmp158 = load float, ptr addrspace(3) %tmp157, align 4
  %tmp159 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 91
  %tmp160 = load float, ptr addrspace(3) %tmp159, align 4
  %tmp161 = tail call float @llvm.fmuladd.f32(float %tmp156, float %tmp158, float %tmp160)
  %tmp162 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 93
  %tmp163 = load float, ptr addrspace(3) %tmp162, align 4
  %tmp164 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 94
  %tmp165 = load float, ptr addrspace(3) %tmp164, align 4
  %tmp166 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 95
  %tmp167 = load float, ptr addrspace(3) %tmp166, align 4
  %tmp168 = tail call float @llvm.fmuladd.f32(float %tmp163, float %tmp165, float %tmp167)
  %tmp169 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 97
  %tmp170 = load float, ptr addrspace(3) %tmp169, align 4
  %tmp171 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 98
  %tmp172 = load float, ptr addrspace(3) %tmp171, align 4
  %tmp173 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 99
  %tmp174 = load float, ptr addrspace(3) %tmp173, align 4
  %tmp175 = tail call float @llvm.fmuladd.f32(float %tmp170, float %tmp172, float %tmp174)
  %tmp176 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 101
  %tmp177 = load float, ptr addrspace(3) %tmp176, align 4
  %tmp178 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 102
  %tmp179 = load float, ptr addrspace(3) %tmp178, align 4
  %tmp180 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 103
  %tmp181 = load float, ptr addrspace(3) %tmp180, align 4
  %tmp182 = tail call float @llvm.fmuladd.f32(float %tmp177, float %tmp179, float %tmp181)
  %tmp183 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 105
  %tmp184 = load float, ptr addrspace(3) %tmp183, align 4
  %tmp185 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 106
  %tmp186 = load float, ptr addrspace(3) %tmp185, align 4
  %tmp187 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 107
  %tmp188 = load float, ptr addrspace(3) %tmp187, align 4
  %tmp189 = tail call float @llvm.fmuladd.f32(float %tmp184, float %tmp186, float %tmp188)
  %tmp190 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 109
  %tmp191 = load float, ptr addrspace(3) %tmp190, align 4
  %tmp192 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 110
  %tmp193 = load float, ptr addrspace(3) %tmp192, align 4
  %tmp194 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 111
  %tmp195 = load float, ptr addrspace(3) %tmp194, align 4
  %tmp196 = tail call float @llvm.fmuladd.f32(float %tmp191, float %tmp193, float %tmp195)
  %tmp197 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 113
  %tmp198 = load float, ptr addrspace(3) %tmp197, align 4
  %tmp199 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 114
  %tmp200 = load float, ptr addrspace(3) %tmp199, align 4
  %tmp201 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 115
  %tmp202 = load float, ptr addrspace(3) %tmp201, align 4
  %tmp203 = tail call float @llvm.fmuladd.f32(float %tmp198, float %tmp200, float %tmp202)
  %tmp204 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 117
  %tmp205 = load float, ptr addrspace(3) %tmp204, align 4
  %tmp206 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 118
  %tmp207 = load float, ptr addrspace(3) %tmp206, align 4
  %tmp208 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 119
  %tmp209 = load float, ptr addrspace(3) %tmp208, align 4
  %tmp210 = tail call float @llvm.fmuladd.f32(float %tmp205, float %tmp207, float %tmp209)
  store float %tmp7, ptr addrspace(1) %arg1, align 4
  %tmp449 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 1
  store float %tmp14, ptr addrspace(1) %tmp449, align 4
  %tmp450 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 2
  store float %tmp21, ptr addrspace(1) %tmp450, align 4
  %tmp451 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 3
  store float %tmp28, ptr addrspace(1) %tmp451, align 4
  %tmp452 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 4
  store float %tmp35, ptr addrspace(1) %tmp452, align 4
  %tmp453 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 5
  store float %tmp42, ptr addrspace(1) %tmp453, align 4
  %tmp454 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 6
  store float %tmp49, ptr addrspace(1) %tmp454, align 4
  %tmp455 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 7
  store float %tmp56, ptr addrspace(1) %tmp455, align 4
  %tmp456 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 8
  store float %tmp63, ptr addrspace(1) %tmp456, align 4
  %tmp457 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 9
  store float %tmp70, ptr addrspace(1) %tmp457, align 4
  %tmp458 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 10
  store float %tmp77, ptr addrspace(1) %tmp458, align 4
  %tmp459 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 11
  store float %tmp84, ptr addrspace(1) %tmp459, align 4
  %tmp460 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 12
  store float %tmp91, ptr addrspace(1) %tmp460, align 4
  %tmp461 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 13
  store float %tmp98, ptr addrspace(1) %tmp461, align 4
  %tmp462 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 14
  store float %tmp105, ptr addrspace(1) %tmp462, align 4
  %tmp463 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 15
  store float %tmp112, ptr addrspace(1) %tmp463, align 4
  %tmp464 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 16
  store float %tmp119, ptr addrspace(1) %tmp464, align 4
  %tmp465 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 17
  store float %tmp126, ptr addrspace(1) %tmp465, align 4
  %tmp466 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 18
  store float %tmp133, ptr addrspace(1) %tmp466, align 4
  %tmp467 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 19
  store float %tmp140, ptr addrspace(1) %tmp467, align 4
  %tmp468 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 20
  store float %tmp147, ptr addrspace(1) %tmp468, align 4
  %tmp469 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 21
  store float %tmp154, ptr addrspace(1) %tmp469, align 4
  %tmp470 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 22
  store float %tmp161, ptr addrspace(1) %tmp470, align 4
  %tmp471 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 23
  store float %tmp168, ptr addrspace(1) %tmp471, align 4
  %tmp472 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 24
  store float %tmp175, ptr addrspace(1) %tmp472, align 4
  %tmp473 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 25
  store float %tmp182, ptr addrspace(1) %tmp473, align 4
  %tmp474 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 26
  store float %tmp189, ptr addrspace(1) %tmp474, align 4
  %tmp475 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 27
  store float %tmp196, ptr addrspace(1) %tmp475, align 4
  %tmp476 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 28
  store float %tmp203, ptr addrspace(1) %tmp476, align 4
  %tmp477 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 29
  store float %tmp210, ptr addrspace(1) %tmp477, align 4
  br i1 %cc, label %bb1, label %bb2

bb2:
  call void asm sideeffect "; use $0","{v[0:31]}"(<32 x i32> %vgpr0)
  call void asm sideeffect "; use $0","{v[0:31]}"(<32 x i32> %vgpr1)
  call void asm sideeffect "; use $0","{v[0:31]}"(<32 x i32> %vgpr2)
  call void asm sideeffect "; use $0","{v[0:31]}"(<32 x i32> %vgpr3)
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float)
