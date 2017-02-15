; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; We expect a two digit VGPR usage here, not a three digit.
; CHECK: NumVgprs: {{[0-9][0-9]$}}

define void @load_fma_store(float addrspace(3)* nocapture readonly %arg, float addrspace(1)* nocapture %arg1) {
bb:
  %tmp = getelementptr inbounds float, float addrspace(3)* %arg, i32 1
  %tmp2 = load float, float addrspace(3)* %tmp, align 4
  %tmp3 = getelementptr inbounds float, float addrspace(3)* %arg, i32 2
  %tmp4 = load float, float addrspace(3)* %tmp3, align 4
  %tmp5 = getelementptr inbounds float, float addrspace(3)* %arg, i32 3
  %tmp6 = load float, float addrspace(3)* %tmp5, align 4
  %tmp7 = tail call float @llvm.fmuladd.f32(float %tmp2, float %tmp4, float %tmp6)
  %tmp8 = getelementptr inbounds float, float addrspace(3)* %arg, i32 5
  %tmp9 = load float, float addrspace(3)* %tmp8, align 4
  %tmp10 = getelementptr inbounds float, float addrspace(3)* %arg, i32 6
  %tmp11 = load float, float addrspace(3)* %tmp10, align 4
  %tmp12 = getelementptr inbounds float, float addrspace(3)* %arg, i32 7
  %tmp13 = load float, float addrspace(3)* %tmp12, align 4
  %tmp14 = tail call float @llvm.fmuladd.f32(float %tmp9, float %tmp11, float %tmp13)
  %tmp15 = getelementptr inbounds float, float addrspace(3)* %arg, i32 9
  %tmp16 = load float, float addrspace(3)* %tmp15, align 4
  %tmp17 = getelementptr inbounds float, float addrspace(3)* %arg, i32 10
  %tmp18 = load float, float addrspace(3)* %tmp17, align 4
  %tmp19 = getelementptr inbounds float, float addrspace(3)* %arg, i32 11
  %tmp20 = load float, float addrspace(3)* %tmp19, align 4
  %tmp21 = tail call float @llvm.fmuladd.f32(float %tmp16, float %tmp18, float %tmp20)
  %tmp22 = getelementptr inbounds float, float addrspace(3)* %arg, i32 13
  %tmp23 = load float, float addrspace(3)* %tmp22, align 4
  %tmp24 = getelementptr inbounds float, float addrspace(3)* %arg, i32 14
  %tmp25 = load float, float addrspace(3)* %tmp24, align 4
  %tmp26 = getelementptr inbounds float, float addrspace(3)* %arg, i32 15
  %tmp27 = load float, float addrspace(3)* %tmp26, align 4
  %tmp28 = tail call float @llvm.fmuladd.f32(float %tmp23, float %tmp25, float %tmp27)
  %tmp29 = getelementptr inbounds float, float addrspace(3)* %arg, i32 17
  %tmp30 = load float, float addrspace(3)* %tmp29, align 4
  %tmp31 = getelementptr inbounds float, float addrspace(3)* %arg, i32 18
  %tmp32 = load float, float addrspace(3)* %tmp31, align 4
  %tmp33 = getelementptr inbounds float, float addrspace(3)* %arg, i32 19
  %tmp34 = load float, float addrspace(3)* %tmp33, align 4
  %tmp35 = tail call float @llvm.fmuladd.f32(float %tmp30, float %tmp32, float %tmp34)
  %tmp36 = getelementptr inbounds float, float addrspace(3)* %arg, i32 21
  %tmp37 = load float, float addrspace(3)* %tmp36, align 4
  %tmp38 = getelementptr inbounds float, float addrspace(3)* %arg, i32 22
  %tmp39 = load float, float addrspace(3)* %tmp38, align 4
  %tmp40 = getelementptr inbounds float, float addrspace(3)* %arg, i32 23
  %tmp41 = load float, float addrspace(3)* %tmp40, align 4
  %tmp42 = tail call float @llvm.fmuladd.f32(float %tmp37, float %tmp39, float %tmp41)
  %tmp43 = getelementptr inbounds float, float addrspace(3)* %arg, i32 25
  %tmp44 = load float, float addrspace(3)* %tmp43, align 4
  %tmp45 = getelementptr inbounds float, float addrspace(3)* %arg, i32 26
  %tmp46 = load float, float addrspace(3)* %tmp45, align 4
  %tmp47 = getelementptr inbounds float, float addrspace(3)* %arg, i32 27
  %tmp48 = load float, float addrspace(3)* %tmp47, align 4
  %tmp49 = tail call float @llvm.fmuladd.f32(float %tmp44, float %tmp46, float %tmp48)
  %tmp50 = getelementptr inbounds float, float addrspace(3)* %arg, i32 29
  %tmp51 = load float, float addrspace(3)* %tmp50, align 4
  %tmp52 = getelementptr inbounds float, float addrspace(3)* %arg, i32 30
  %tmp53 = load float, float addrspace(3)* %tmp52, align 4
  %tmp54 = getelementptr inbounds float, float addrspace(3)* %arg, i32 31
  %tmp55 = load float, float addrspace(3)* %tmp54, align 4
  %tmp56 = tail call float @llvm.fmuladd.f32(float %tmp51, float %tmp53, float %tmp55)
  %tmp57 = getelementptr inbounds float, float addrspace(3)* %arg, i32 33
  %tmp58 = load float, float addrspace(3)* %tmp57, align 4
  %tmp59 = getelementptr inbounds float, float addrspace(3)* %arg, i32 34
  %tmp60 = load float, float addrspace(3)* %tmp59, align 4
  %tmp61 = getelementptr inbounds float, float addrspace(3)* %arg, i32 35
  %tmp62 = load float, float addrspace(3)* %tmp61, align 4
  %tmp63 = tail call float @llvm.fmuladd.f32(float %tmp58, float %tmp60, float %tmp62)
  %tmp64 = getelementptr inbounds float, float addrspace(3)* %arg, i32 37
  %tmp65 = load float, float addrspace(3)* %tmp64, align 4
  %tmp66 = getelementptr inbounds float, float addrspace(3)* %arg, i32 38
  %tmp67 = load float, float addrspace(3)* %tmp66, align 4
  %tmp68 = getelementptr inbounds float, float addrspace(3)* %arg, i32 39
  %tmp69 = load float, float addrspace(3)* %tmp68, align 4
  %tmp70 = tail call float @llvm.fmuladd.f32(float %tmp65, float %tmp67, float %tmp69)
  %tmp71 = getelementptr inbounds float, float addrspace(3)* %arg, i32 41
  %tmp72 = load float, float addrspace(3)* %tmp71, align 4
  %tmp73 = getelementptr inbounds float, float addrspace(3)* %arg, i32 42
  %tmp74 = load float, float addrspace(3)* %tmp73, align 4
  %tmp75 = getelementptr inbounds float, float addrspace(3)* %arg, i32 43
  %tmp76 = load float, float addrspace(3)* %tmp75, align 4
  %tmp77 = tail call float @llvm.fmuladd.f32(float %tmp72, float %tmp74, float %tmp76)
  %tmp78 = getelementptr inbounds float, float addrspace(3)* %arg, i32 45
  %tmp79 = load float, float addrspace(3)* %tmp78, align 4
  %tmp80 = getelementptr inbounds float, float addrspace(3)* %arg, i32 46
  %tmp81 = load float, float addrspace(3)* %tmp80, align 4
  %tmp82 = getelementptr inbounds float, float addrspace(3)* %arg, i32 47
  %tmp83 = load float, float addrspace(3)* %tmp82, align 4
  %tmp84 = tail call float @llvm.fmuladd.f32(float %tmp79, float %tmp81, float %tmp83)
  %tmp85 = getelementptr inbounds float, float addrspace(3)* %arg, i32 49
  %tmp86 = load float, float addrspace(3)* %tmp85, align 4
  %tmp87 = getelementptr inbounds float, float addrspace(3)* %arg, i32 50
  %tmp88 = load float, float addrspace(3)* %tmp87, align 4
  %tmp89 = getelementptr inbounds float, float addrspace(3)* %arg, i32 51
  %tmp90 = load float, float addrspace(3)* %tmp89, align 4
  %tmp91 = tail call float @llvm.fmuladd.f32(float %tmp86, float %tmp88, float %tmp90)
  %tmp92 = getelementptr inbounds float, float addrspace(3)* %arg, i32 53
  %tmp93 = load float, float addrspace(3)* %tmp92, align 4
  %tmp94 = getelementptr inbounds float, float addrspace(3)* %arg, i32 54
  %tmp95 = load float, float addrspace(3)* %tmp94, align 4
  %tmp96 = getelementptr inbounds float, float addrspace(3)* %arg, i32 55
  %tmp97 = load float, float addrspace(3)* %tmp96, align 4
  %tmp98 = tail call float @llvm.fmuladd.f32(float %tmp93, float %tmp95, float %tmp97)
  %tmp99 = getelementptr inbounds float, float addrspace(3)* %arg, i32 57
  %tmp100 = load float, float addrspace(3)* %tmp99, align 4
  %tmp101 = getelementptr inbounds float, float addrspace(3)* %arg, i32 58
  %tmp102 = load float, float addrspace(3)* %tmp101, align 4
  %tmp103 = getelementptr inbounds float, float addrspace(3)* %arg, i32 59
  %tmp104 = load float, float addrspace(3)* %tmp103, align 4
  %tmp105 = tail call float @llvm.fmuladd.f32(float %tmp100, float %tmp102, float %tmp104)
  %tmp106 = getelementptr inbounds float, float addrspace(3)* %arg, i32 61
  %tmp107 = load float, float addrspace(3)* %tmp106, align 4
  %tmp108 = getelementptr inbounds float, float addrspace(3)* %arg, i32 62
  %tmp109 = load float, float addrspace(3)* %tmp108, align 4
  %tmp110 = getelementptr inbounds float, float addrspace(3)* %arg, i32 63
  %tmp111 = load float, float addrspace(3)* %tmp110, align 4
  %tmp112 = tail call float @llvm.fmuladd.f32(float %tmp107, float %tmp109, float %tmp111)
  %tmp113 = getelementptr inbounds float, float addrspace(3)* %arg, i32 65
  %tmp114 = load float, float addrspace(3)* %tmp113, align 4
  %tmp115 = getelementptr inbounds float, float addrspace(3)* %arg, i32 66
  %tmp116 = load float, float addrspace(3)* %tmp115, align 4
  %tmp117 = getelementptr inbounds float, float addrspace(3)* %arg, i32 67
  %tmp118 = load float, float addrspace(3)* %tmp117, align 4
  %tmp119 = tail call float @llvm.fmuladd.f32(float %tmp114, float %tmp116, float %tmp118)
  %tmp120 = getelementptr inbounds float, float addrspace(3)* %arg, i32 69
  %tmp121 = load float, float addrspace(3)* %tmp120, align 4
  %tmp122 = getelementptr inbounds float, float addrspace(3)* %arg, i32 70
  %tmp123 = load float, float addrspace(3)* %tmp122, align 4
  %tmp124 = getelementptr inbounds float, float addrspace(3)* %arg, i32 71
  %tmp125 = load float, float addrspace(3)* %tmp124, align 4
  %tmp126 = tail call float @llvm.fmuladd.f32(float %tmp121, float %tmp123, float %tmp125)
  %tmp127 = getelementptr inbounds float, float addrspace(3)* %arg, i32 73
  %tmp128 = load float, float addrspace(3)* %tmp127, align 4
  %tmp129 = getelementptr inbounds float, float addrspace(3)* %arg, i32 74
  %tmp130 = load float, float addrspace(3)* %tmp129, align 4
  %tmp131 = getelementptr inbounds float, float addrspace(3)* %arg, i32 75
  %tmp132 = load float, float addrspace(3)* %tmp131, align 4
  %tmp133 = tail call float @llvm.fmuladd.f32(float %tmp128, float %tmp130, float %tmp132)
  %tmp134 = getelementptr inbounds float, float addrspace(3)* %arg, i32 77
  %tmp135 = load float, float addrspace(3)* %tmp134, align 4
  %tmp136 = getelementptr inbounds float, float addrspace(3)* %arg, i32 78
  %tmp137 = load float, float addrspace(3)* %tmp136, align 4
  %tmp138 = getelementptr inbounds float, float addrspace(3)* %arg, i32 79
  %tmp139 = load float, float addrspace(3)* %tmp138, align 4
  %tmp140 = tail call float @llvm.fmuladd.f32(float %tmp135, float %tmp137, float %tmp139)
  %tmp141 = getelementptr inbounds float, float addrspace(3)* %arg, i32 81
  %tmp142 = load float, float addrspace(3)* %tmp141, align 4
  %tmp143 = getelementptr inbounds float, float addrspace(3)* %arg, i32 82
  %tmp144 = load float, float addrspace(3)* %tmp143, align 4
  %tmp145 = getelementptr inbounds float, float addrspace(3)* %arg, i32 83
  %tmp146 = load float, float addrspace(3)* %tmp145, align 4
  %tmp147 = tail call float @llvm.fmuladd.f32(float %tmp142, float %tmp144, float %tmp146)
  %tmp148 = getelementptr inbounds float, float addrspace(3)* %arg, i32 85
  %tmp149 = load float, float addrspace(3)* %tmp148, align 4
  %tmp150 = getelementptr inbounds float, float addrspace(3)* %arg, i32 86
  %tmp151 = load float, float addrspace(3)* %tmp150, align 4
  %tmp152 = getelementptr inbounds float, float addrspace(3)* %arg, i32 87
  %tmp153 = load float, float addrspace(3)* %tmp152, align 4
  %tmp154 = tail call float @llvm.fmuladd.f32(float %tmp149, float %tmp151, float %tmp153)
  %tmp155 = getelementptr inbounds float, float addrspace(3)* %arg, i32 89
  %tmp156 = load float, float addrspace(3)* %tmp155, align 4
  %tmp157 = getelementptr inbounds float, float addrspace(3)* %arg, i32 90
  %tmp158 = load float, float addrspace(3)* %tmp157, align 4
  %tmp159 = getelementptr inbounds float, float addrspace(3)* %arg, i32 91
  %tmp160 = load float, float addrspace(3)* %tmp159, align 4
  %tmp161 = tail call float @llvm.fmuladd.f32(float %tmp156, float %tmp158, float %tmp160)
  %tmp162 = getelementptr inbounds float, float addrspace(3)* %arg, i32 93
  %tmp163 = load float, float addrspace(3)* %tmp162, align 4
  %tmp164 = getelementptr inbounds float, float addrspace(3)* %arg, i32 94
  %tmp165 = load float, float addrspace(3)* %tmp164, align 4
  %tmp166 = getelementptr inbounds float, float addrspace(3)* %arg, i32 95
  %tmp167 = load float, float addrspace(3)* %tmp166, align 4
  %tmp168 = tail call float @llvm.fmuladd.f32(float %tmp163, float %tmp165, float %tmp167)
  %tmp169 = getelementptr inbounds float, float addrspace(3)* %arg, i32 97
  %tmp170 = load float, float addrspace(3)* %tmp169, align 4
  %tmp171 = getelementptr inbounds float, float addrspace(3)* %arg, i32 98
  %tmp172 = load float, float addrspace(3)* %tmp171, align 4
  %tmp173 = getelementptr inbounds float, float addrspace(3)* %arg, i32 99
  %tmp174 = load float, float addrspace(3)* %tmp173, align 4
  %tmp175 = tail call float @llvm.fmuladd.f32(float %tmp170, float %tmp172, float %tmp174)
  %tmp176 = getelementptr inbounds float, float addrspace(3)* %arg, i32 101
  %tmp177 = load float, float addrspace(3)* %tmp176, align 4
  %tmp178 = getelementptr inbounds float, float addrspace(3)* %arg, i32 102
  %tmp179 = load float, float addrspace(3)* %tmp178, align 4
  %tmp180 = getelementptr inbounds float, float addrspace(3)* %arg, i32 103
  %tmp181 = load float, float addrspace(3)* %tmp180, align 4
  %tmp182 = tail call float @llvm.fmuladd.f32(float %tmp177, float %tmp179, float %tmp181)
  %tmp183 = getelementptr inbounds float, float addrspace(3)* %arg, i32 105
  %tmp184 = load float, float addrspace(3)* %tmp183, align 4
  %tmp185 = getelementptr inbounds float, float addrspace(3)* %arg, i32 106
  %tmp186 = load float, float addrspace(3)* %tmp185, align 4
  %tmp187 = getelementptr inbounds float, float addrspace(3)* %arg, i32 107
  %tmp188 = load float, float addrspace(3)* %tmp187, align 4
  %tmp189 = tail call float @llvm.fmuladd.f32(float %tmp184, float %tmp186, float %tmp188)
  %tmp190 = getelementptr inbounds float, float addrspace(3)* %arg, i32 109
  %tmp191 = load float, float addrspace(3)* %tmp190, align 4
  %tmp192 = getelementptr inbounds float, float addrspace(3)* %arg, i32 110
  %tmp193 = load float, float addrspace(3)* %tmp192, align 4
  %tmp194 = getelementptr inbounds float, float addrspace(3)* %arg, i32 111
  %tmp195 = load float, float addrspace(3)* %tmp194, align 4
  %tmp196 = tail call float @llvm.fmuladd.f32(float %tmp191, float %tmp193, float %tmp195)
  %tmp197 = getelementptr inbounds float, float addrspace(3)* %arg, i32 113
  %tmp198 = load float, float addrspace(3)* %tmp197, align 4
  %tmp199 = getelementptr inbounds float, float addrspace(3)* %arg, i32 114
  %tmp200 = load float, float addrspace(3)* %tmp199, align 4
  %tmp201 = getelementptr inbounds float, float addrspace(3)* %arg, i32 115
  %tmp202 = load float, float addrspace(3)* %tmp201, align 4
  %tmp203 = tail call float @llvm.fmuladd.f32(float %tmp198, float %tmp200, float %tmp202)
  %tmp204 = getelementptr inbounds float, float addrspace(3)* %arg, i32 117
  %tmp205 = load float, float addrspace(3)* %tmp204, align 4
  %tmp206 = getelementptr inbounds float, float addrspace(3)* %arg, i32 118
  %tmp207 = load float, float addrspace(3)* %tmp206, align 4
  %tmp208 = getelementptr inbounds float, float addrspace(3)* %arg, i32 119
  %tmp209 = load float, float addrspace(3)* %tmp208, align 4
  %tmp210 = tail call float @llvm.fmuladd.f32(float %tmp205, float %tmp207, float %tmp209)
  %tmp211 = getelementptr inbounds float, float addrspace(3)* %arg, i32 121
  %tmp212 = load float, float addrspace(3)* %tmp211, align 4
  %tmp213 = getelementptr inbounds float, float addrspace(3)* %arg, i32 122
  %tmp214 = load float, float addrspace(3)* %tmp213, align 4
  %tmp215 = getelementptr inbounds float, float addrspace(3)* %arg, i32 123
  %tmp216 = load float, float addrspace(3)* %tmp215, align 4
  %tmp217 = tail call float @llvm.fmuladd.f32(float %tmp212, float %tmp214, float %tmp216)
  %tmp218 = getelementptr inbounds float, float addrspace(3)* %arg, i32 125
  %tmp219 = load float, float addrspace(3)* %tmp218, align 4
  %tmp220 = getelementptr inbounds float, float addrspace(3)* %arg, i32 126
  %tmp221 = load float, float addrspace(3)* %tmp220, align 4
  %tmp222 = getelementptr inbounds float, float addrspace(3)* %arg, i32 127
  %tmp223 = load float, float addrspace(3)* %tmp222, align 4
  %tmp224 = tail call float @llvm.fmuladd.f32(float %tmp219, float %tmp221, float %tmp223)
  %tmp225 = getelementptr inbounds float, float addrspace(3)* %arg, i32 129
  %tmp226 = load float, float addrspace(3)* %tmp225, align 4
  %tmp227 = getelementptr inbounds float, float addrspace(3)* %arg, i32 130
  %tmp228 = load float, float addrspace(3)* %tmp227, align 4
  %tmp229 = getelementptr inbounds float, float addrspace(3)* %arg, i32 131
  %tmp230 = load float, float addrspace(3)* %tmp229, align 4
  %tmp231 = tail call float @llvm.fmuladd.f32(float %tmp226, float %tmp228, float %tmp230)
  %tmp232 = getelementptr inbounds float, float addrspace(3)* %arg, i32 133
  %tmp233 = load float, float addrspace(3)* %tmp232, align 4
  %tmp234 = getelementptr inbounds float, float addrspace(3)* %arg, i32 134
  %tmp235 = load float, float addrspace(3)* %tmp234, align 4
  %tmp236 = getelementptr inbounds float, float addrspace(3)* %arg, i32 135
  %tmp237 = load float, float addrspace(3)* %tmp236, align 4
  %tmp238 = tail call float @llvm.fmuladd.f32(float %tmp233, float %tmp235, float %tmp237)
  %tmp239 = getelementptr inbounds float, float addrspace(3)* %arg, i32 137
  %tmp240 = load float, float addrspace(3)* %tmp239, align 4
  %tmp241 = getelementptr inbounds float, float addrspace(3)* %arg, i32 138
  %tmp242 = load float, float addrspace(3)* %tmp241, align 4
  %tmp243 = getelementptr inbounds float, float addrspace(3)* %arg, i32 139
  %tmp244 = load float, float addrspace(3)* %tmp243, align 4
  %tmp245 = tail call float @llvm.fmuladd.f32(float %tmp240, float %tmp242, float %tmp244)
  %tmp246 = getelementptr inbounds float, float addrspace(3)* %arg, i32 141
  %tmp247 = load float, float addrspace(3)* %tmp246, align 4
  %tmp248 = getelementptr inbounds float, float addrspace(3)* %arg, i32 142
  %tmp249 = load float, float addrspace(3)* %tmp248, align 4
  %tmp250 = getelementptr inbounds float, float addrspace(3)* %arg, i32 143
  %tmp251 = load float, float addrspace(3)* %tmp250, align 4
  %tmp252 = tail call float @llvm.fmuladd.f32(float %tmp247, float %tmp249, float %tmp251)
  %tmp253 = getelementptr inbounds float, float addrspace(3)* %arg, i32 145
  %tmp254 = load float, float addrspace(3)* %tmp253, align 4
  %tmp255 = getelementptr inbounds float, float addrspace(3)* %arg, i32 146
  %tmp256 = load float, float addrspace(3)* %tmp255, align 4
  %tmp257 = getelementptr inbounds float, float addrspace(3)* %arg, i32 147
  %tmp258 = load float, float addrspace(3)* %tmp257, align 4
  %tmp259 = tail call float @llvm.fmuladd.f32(float %tmp254, float %tmp256, float %tmp258)
  %tmp260 = getelementptr inbounds float, float addrspace(3)* %arg, i32 149
  %tmp261 = load float, float addrspace(3)* %tmp260, align 4
  %tmp262 = getelementptr inbounds float, float addrspace(3)* %arg, i32 150
  %tmp263 = load float, float addrspace(3)* %tmp262, align 4
  %tmp264 = getelementptr inbounds float, float addrspace(3)* %arg, i32 151
  %tmp265 = load float, float addrspace(3)* %tmp264, align 4
  %tmp266 = tail call float @llvm.fmuladd.f32(float %tmp261, float %tmp263, float %tmp265)
  %tmp267 = getelementptr inbounds float, float addrspace(3)* %arg, i32 153
  %tmp268 = load float, float addrspace(3)* %tmp267, align 4
  %tmp269 = getelementptr inbounds float, float addrspace(3)* %arg, i32 154
  %tmp270 = load float, float addrspace(3)* %tmp269, align 4
  %tmp271 = getelementptr inbounds float, float addrspace(3)* %arg, i32 155
  %tmp272 = load float, float addrspace(3)* %tmp271, align 4
  %tmp273 = tail call float @llvm.fmuladd.f32(float %tmp268, float %tmp270, float %tmp272)
  %tmp274 = getelementptr inbounds float, float addrspace(3)* %arg, i32 157
  %tmp275 = load float, float addrspace(3)* %tmp274, align 4
  %tmp276 = getelementptr inbounds float, float addrspace(3)* %arg, i32 158
  %tmp277 = load float, float addrspace(3)* %tmp276, align 4
  %tmp278 = getelementptr inbounds float, float addrspace(3)* %arg, i32 159
  %tmp279 = load float, float addrspace(3)* %tmp278, align 4
  %tmp280 = tail call float @llvm.fmuladd.f32(float %tmp275, float %tmp277, float %tmp279)
  %tmp281 = getelementptr inbounds float, float addrspace(3)* %arg, i32 161
  %tmp282 = load float, float addrspace(3)* %tmp281, align 4
  %tmp283 = getelementptr inbounds float, float addrspace(3)* %arg, i32 162
  %tmp284 = load float, float addrspace(3)* %tmp283, align 4
  %tmp285 = getelementptr inbounds float, float addrspace(3)* %arg, i32 163
  %tmp286 = load float, float addrspace(3)* %tmp285, align 4
  %tmp287 = tail call float @llvm.fmuladd.f32(float %tmp282, float %tmp284, float %tmp286)
  %tmp288 = getelementptr inbounds float, float addrspace(3)* %arg, i32 165
  %tmp289 = load float, float addrspace(3)* %tmp288, align 4
  %tmp290 = getelementptr inbounds float, float addrspace(3)* %arg, i32 166
  %tmp291 = load float, float addrspace(3)* %tmp290, align 4
  %tmp292 = getelementptr inbounds float, float addrspace(3)* %arg, i32 167
  %tmp293 = load float, float addrspace(3)* %tmp292, align 4
  %tmp294 = tail call float @llvm.fmuladd.f32(float %tmp289, float %tmp291, float %tmp293)
  %tmp295 = getelementptr inbounds float, float addrspace(3)* %arg, i32 169
  %tmp296 = load float, float addrspace(3)* %tmp295, align 4
  %tmp297 = getelementptr inbounds float, float addrspace(3)* %arg, i32 170
  %tmp298 = load float, float addrspace(3)* %tmp297, align 4
  %tmp299 = getelementptr inbounds float, float addrspace(3)* %arg, i32 171
  %tmp300 = load float, float addrspace(3)* %tmp299, align 4
  %tmp301 = tail call float @llvm.fmuladd.f32(float %tmp296, float %tmp298, float %tmp300)
  %tmp302 = getelementptr inbounds float, float addrspace(3)* %arg, i32 173
  %tmp303 = load float, float addrspace(3)* %tmp302, align 4
  %tmp304 = getelementptr inbounds float, float addrspace(3)* %arg, i32 174
  %tmp305 = load float, float addrspace(3)* %tmp304, align 4
  %tmp306 = getelementptr inbounds float, float addrspace(3)* %arg, i32 175
  %tmp307 = load float, float addrspace(3)* %tmp306, align 4
  %tmp308 = tail call float @llvm.fmuladd.f32(float %tmp303, float %tmp305, float %tmp307)
  %tmp309 = getelementptr inbounds float, float addrspace(3)* %arg, i32 177
  %tmp310 = load float, float addrspace(3)* %tmp309, align 4
  %tmp311 = getelementptr inbounds float, float addrspace(3)* %arg, i32 178
  %tmp312 = load float, float addrspace(3)* %tmp311, align 4
  %tmp313 = getelementptr inbounds float, float addrspace(3)* %arg, i32 179
  %tmp314 = load float, float addrspace(3)* %tmp313, align 4
  %tmp315 = tail call float @llvm.fmuladd.f32(float %tmp310, float %tmp312, float %tmp314)
  %tmp316 = getelementptr inbounds float, float addrspace(3)* %arg, i32 181
  %tmp317 = load float, float addrspace(3)* %tmp316, align 4
  %tmp318 = getelementptr inbounds float, float addrspace(3)* %arg, i32 182
  %tmp319 = load float, float addrspace(3)* %tmp318, align 4
  %tmp320 = getelementptr inbounds float, float addrspace(3)* %arg, i32 183
  %tmp321 = load float, float addrspace(3)* %tmp320, align 4
  %tmp322 = tail call float @llvm.fmuladd.f32(float %tmp317, float %tmp319, float %tmp321)
  %tmp323 = getelementptr inbounds float, float addrspace(3)* %arg, i32 185
  %tmp324 = load float, float addrspace(3)* %tmp323, align 4
  %tmp325 = getelementptr inbounds float, float addrspace(3)* %arg, i32 186
  %tmp326 = load float, float addrspace(3)* %tmp325, align 4
  %tmp327 = getelementptr inbounds float, float addrspace(3)* %arg, i32 187
  %tmp328 = load float, float addrspace(3)* %tmp327, align 4
  %tmp329 = tail call float @llvm.fmuladd.f32(float %tmp324, float %tmp326, float %tmp328)
  %tmp330 = getelementptr inbounds float, float addrspace(3)* %arg, i32 189
  %tmp331 = load float, float addrspace(3)* %tmp330, align 4
  %tmp332 = getelementptr inbounds float, float addrspace(3)* %arg, i32 190
  %tmp333 = load float, float addrspace(3)* %tmp332, align 4
  %tmp334 = getelementptr inbounds float, float addrspace(3)* %arg, i32 191
  %tmp335 = load float, float addrspace(3)* %tmp334, align 4
  %tmp336 = tail call float @llvm.fmuladd.f32(float %tmp331, float %tmp333, float %tmp335)
  %tmp337 = getelementptr inbounds float, float addrspace(3)* %arg, i32 193
  %tmp338 = load float, float addrspace(3)* %tmp337, align 4
  %tmp339 = getelementptr inbounds float, float addrspace(3)* %arg, i32 194
  %tmp340 = load float, float addrspace(3)* %tmp339, align 4
  %tmp341 = getelementptr inbounds float, float addrspace(3)* %arg, i32 195
  %tmp342 = load float, float addrspace(3)* %tmp341, align 4
  %tmp343 = tail call float @llvm.fmuladd.f32(float %tmp338, float %tmp340, float %tmp342)
  %tmp344 = getelementptr inbounds float, float addrspace(3)* %arg, i32 197
  %tmp345 = load float, float addrspace(3)* %tmp344, align 4
  %tmp346 = getelementptr inbounds float, float addrspace(3)* %arg, i32 198
  %tmp347 = load float, float addrspace(3)* %tmp346, align 4
  %tmp348 = getelementptr inbounds float, float addrspace(3)* %arg, i32 199
  %tmp349 = load float, float addrspace(3)* %tmp348, align 4
  %tmp350 = tail call float @llvm.fmuladd.f32(float %tmp345, float %tmp347, float %tmp349)
  %tmp351 = getelementptr inbounds float, float addrspace(3)* %arg, i32 201
  %tmp352 = load float, float addrspace(3)* %tmp351, align 4
  %tmp353 = getelementptr inbounds float, float addrspace(3)* %arg, i32 202
  %tmp354 = load float, float addrspace(3)* %tmp353, align 4
  %tmp355 = getelementptr inbounds float, float addrspace(3)* %arg, i32 203
  %tmp356 = load float, float addrspace(3)* %tmp355, align 4
  %tmp357 = tail call float @llvm.fmuladd.f32(float %tmp352, float %tmp354, float %tmp356)
  %tmp358 = getelementptr inbounds float, float addrspace(3)* %arg, i32 205
  %tmp359 = load float, float addrspace(3)* %tmp358, align 4
  %tmp360 = getelementptr inbounds float, float addrspace(3)* %arg, i32 206
  %tmp361 = load float, float addrspace(3)* %tmp360, align 4
  %tmp362 = getelementptr inbounds float, float addrspace(3)* %arg, i32 207
  %tmp363 = load float, float addrspace(3)* %tmp362, align 4
  %tmp364 = tail call float @llvm.fmuladd.f32(float %tmp359, float %tmp361, float %tmp363)
  %tmp365 = getelementptr inbounds float, float addrspace(3)* %arg, i32 209
  %tmp366 = load float, float addrspace(3)* %tmp365, align 4
  %tmp367 = getelementptr inbounds float, float addrspace(3)* %arg, i32 210
  %tmp368 = load float, float addrspace(3)* %tmp367, align 4
  %tmp369 = getelementptr inbounds float, float addrspace(3)* %arg, i32 211
  %tmp370 = load float, float addrspace(3)* %tmp369, align 4
  %tmp371 = tail call float @llvm.fmuladd.f32(float %tmp366, float %tmp368, float %tmp370)
  %tmp372 = getelementptr inbounds float, float addrspace(3)* %arg, i32 213
  %tmp373 = load float, float addrspace(3)* %tmp372, align 4
  %tmp374 = getelementptr inbounds float, float addrspace(3)* %arg, i32 214
  %tmp375 = load float, float addrspace(3)* %tmp374, align 4
  %tmp376 = getelementptr inbounds float, float addrspace(3)* %arg, i32 215
  %tmp377 = load float, float addrspace(3)* %tmp376, align 4
  %tmp378 = tail call float @llvm.fmuladd.f32(float %tmp373, float %tmp375, float %tmp377)
  %tmp379 = getelementptr inbounds float, float addrspace(3)* %arg, i32 217
  %tmp380 = load float, float addrspace(3)* %tmp379, align 4
  %tmp381 = getelementptr inbounds float, float addrspace(3)* %arg, i32 218
  %tmp382 = load float, float addrspace(3)* %tmp381, align 4
  %tmp383 = getelementptr inbounds float, float addrspace(3)* %arg, i32 219
  %tmp384 = load float, float addrspace(3)* %tmp383, align 4
  %tmp385 = tail call float @llvm.fmuladd.f32(float %tmp380, float %tmp382, float %tmp384)
  %tmp386 = getelementptr inbounds float, float addrspace(3)* %arg, i32 221
  %tmp387 = load float, float addrspace(3)* %tmp386, align 4
  %tmp388 = getelementptr inbounds float, float addrspace(3)* %arg, i32 222
  %tmp389 = load float, float addrspace(3)* %tmp388, align 4
  %tmp390 = getelementptr inbounds float, float addrspace(3)* %arg, i32 223
  %tmp391 = load float, float addrspace(3)* %tmp390, align 4
  %tmp392 = tail call float @llvm.fmuladd.f32(float %tmp387, float %tmp389, float %tmp391)
  %tmp393 = getelementptr inbounds float, float addrspace(3)* %arg, i32 225
  %tmp394 = load float, float addrspace(3)* %tmp393, align 4
  %tmp395 = getelementptr inbounds float, float addrspace(3)* %arg, i32 226
  %tmp396 = load float, float addrspace(3)* %tmp395, align 4
  %tmp397 = getelementptr inbounds float, float addrspace(3)* %arg, i32 227
  %tmp398 = load float, float addrspace(3)* %tmp397, align 4
  %tmp399 = tail call float @llvm.fmuladd.f32(float %tmp394, float %tmp396, float %tmp398)
  %tmp400 = getelementptr inbounds float, float addrspace(3)* %arg, i32 229
  %tmp401 = load float, float addrspace(3)* %tmp400, align 4
  %tmp402 = getelementptr inbounds float, float addrspace(3)* %arg, i32 230
  %tmp403 = load float, float addrspace(3)* %tmp402, align 4
  %tmp404 = getelementptr inbounds float, float addrspace(3)* %arg, i32 231
  %tmp405 = load float, float addrspace(3)* %tmp404, align 4
  %tmp406 = tail call float @llvm.fmuladd.f32(float %tmp401, float %tmp403, float %tmp405)
  %tmp407 = getelementptr inbounds float, float addrspace(3)* %arg, i32 233
  %tmp408 = load float, float addrspace(3)* %tmp407, align 4
  %tmp409 = getelementptr inbounds float, float addrspace(3)* %arg, i32 234
  %tmp410 = load float, float addrspace(3)* %tmp409, align 4
  %tmp411 = getelementptr inbounds float, float addrspace(3)* %arg, i32 235
  %tmp412 = load float, float addrspace(3)* %tmp411, align 4
  %tmp413 = tail call float @llvm.fmuladd.f32(float %tmp408, float %tmp410, float %tmp412)
  %tmp414 = getelementptr inbounds float, float addrspace(3)* %arg, i32 237
  %tmp415 = load float, float addrspace(3)* %tmp414, align 4
  %tmp416 = getelementptr inbounds float, float addrspace(3)* %arg, i32 238
  %tmp417 = load float, float addrspace(3)* %tmp416, align 4
  %tmp418 = getelementptr inbounds float, float addrspace(3)* %arg, i32 239
  %tmp419 = load float, float addrspace(3)* %tmp418, align 4
  %tmp420 = tail call float @llvm.fmuladd.f32(float %tmp415, float %tmp417, float %tmp419)
  %tmp421 = getelementptr inbounds float, float addrspace(3)* %arg, i32 241
  %tmp422 = load float, float addrspace(3)* %tmp421, align 4
  %tmp423 = getelementptr inbounds float, float addrspace(3)* %arg, i32 242
  %tmp424 = load float, float addrspace(3)* %tmp423, align 4
  %tmp425 = getelementptr inbounds float, float addrspace(3)* %arg, i32 243
  %tmp426 = load float, float addrspace(3)* %tmp425, align 4
  %tmp427 = tail call float @llvm.fmuladd.f32(float %tmp422, float %tmp424, float %tmp426)
  %tmp428 = getelementptr inbounds float, float addrspace(3)* %arg, i32 245
  %tmp429 = load float, float addrspace(3)* %tmp428, align 4
  %tmp430 = getelementptr inbounds float, float addrspace(3)* %arg, i32 246
  %tmp431 = load float, float addrspace(3)* %tmp430, align 4
  %tmp432 = getelementptr inbounds float, float addrspace(3)* %arg, i32 247
  %tmp433 = load float, float addrspace(3)* %tmp432, align 4
  %tmp434 = tail call float @llvm.fmuladd.f32(float %tmp429, float %tmp431, float %tmp433)
  %tmp435 = getelementptr inbounds float, float addrspace(3)* %arg, i32 249
  %tmp436 = load float, float addrspace(3)* %tmp435, align 4
  %tmp437 = getelementptr inbounds float, float addrspace(3)* %arg, i32 250
  %tmp438 = load float, float addrspace(3)* %tmp437, align 4
  %tmp439 = getelementptr inbounds float, float addrspace(3)* %arg, i32 251
  %tmp440 = load float, float addrspace(3)* %tmp439, align 4
  %tmp441 = tail call float @llvm.fmuladd.f32(float %tmp436, float %tmp438, float %tmp440)
  %tmp442 = getelementptr inbounds float, float addrspace(3)* %arg, i32 253
  %tmp443 = load float, float addrspace(3)* %tmp442, align 4
  %tmp444 = getelementptr inbounds float, float addrspace(3)* %arg, i32 254
  %tmp445 = load float, float addrspace(3)* %tmp444, align 4
  %tmp446 = getelementptr inbounds float, float addrspace(3)* %arg, i32 255
  %tmp447 = load float, float addrspace(3)* %tmp446, align 4
  %tmp448 = tail call float @llvm.fmuladd.f32(float %tmp443, float %tmp445, float %tmp447)
  store float %tmp7, float addrspace(1)* %arg1, align 4
  %tmp449 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 1
  store float %tmp14, float addrspace(1)* %tmp449, align 4
  %tmp450 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 2
  store float %tmp21, float addrspace(1)* %tmp450, align 4
  %tmp451 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 3
  store float %tmp28, float addrspace(1)* %tmp451, align 4
  %tmp452 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 4
  store float %tmp35, float addrspace(1)* %tmp452, align 4
  %tmp453 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 5
  store float %tmp42, float addrspace(1)* %tmp453, align 4
  %tmp454 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 6
  store float %tmp49, float addrspace(1)* %tmp454, align 4
  %tmp455 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 7
  store float %tmp56, float addrspace(1)* %tmp455, align 4
  %tmp456 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 8
  store float %tmp63, float addrspace(1)* %tmp456, align 4
  %tmp457 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 9
  store float %tmp70, float addrspace(1)* %tmp457, align 4
  %tmp458 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 10
  store float %tmp77, float addrspace(1)* %tmp458, align 4
  %tmp459 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 11
  store float %tmp84, float addrspace(1)* %tmp459, align 4
  %tmp460 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 12
  store float %tmp91, float addrspace(1)* %tmp460, align 4
  %tmp461 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 13
  store float %tmp98, float addrspace(1)* %tmp461, align 4
  %tmp462 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 14
  store float %tmp105, float addrspace(1)* %tmp462, align 4
  %tmp463 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 15
  store float %tmp112, float addrspace(1)* %tmp463, align 4
  %tmp464 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 16
  store float %tmp119, float addrspace(1)* %tmp464, align 4
  %tmp465 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 17
  store float %tmp126, float addrspace(1)* %tmp465, align 4
  %tmp466 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 18
  store float %tmp133, float addrspace(1)* %tmp466, align 4
  %tmp467 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 19
  store float %tmp140, float addrspace(1)* %tmp467, align 4
  %tmp468 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 20
  store float %tmp147, float addrspace(1)* %tmp468, align 4
  %tmp469 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 21
  store float %tmp154, float addrspace(1)* %tmp469, align 4
  %tmp470 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 22
  store float %tmp161, float addrspace(1)* %tmp470, align 4
  %tmp471 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 23
  store float %tmp168, float addrspace(1)* %tmp471, align 4
  %tmp472 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 24
  store float %tmp175, float addrspace(1)* %tmp472, align 4
  %tmp473 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 25
  store float %tmp182, float addrspace(1)* %tmp473, align 4
  %tmp474 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 26
  store float %tmp189, float addrspace(1)* %tmp474, align 4
  %tmp475 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 27
  store float %tmp196, float addrspace(1)* %tmp475, align 4
  %tmp476 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 28
  store float %tmp203, float addrspace(1)* %tmp476, align 4
  %tmp477 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 29
  store float %tmp210, float addrspace(1)* %tmp477, align 4
  %tmp478 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 30
  store float %tmp217, float addrspace(1)* %tmp478, align 4
  %tmp479 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 31
  store float %tmp224, float addrspace(1)* %tmp479, align 4
  %tmp480 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 32
  store float %tmp231, float addrspace(1)* %tmp480, align 4
  %tmp481 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 33
  store float %tmp238, float addrspace(1)* %tmp481, align 4
  %tmp482 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 34
  store float %tmp245, float addrspace(1)* %tmp482, align 4
  %tmp483 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 35
  store float %tmp252, float addrspace(1)* %tmp483, align 4
  %tmp484 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 36
  store float %tmp259, float addrspace(1)* %tmp484, align 4
  %tmp485 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 37
  store float %tmp266, float addrspace(1)* %tmp485, align 4
  %tmp486 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 38
  store float %tmp273, float addrspace(1)* %tmp486, align 4
  %tmp487 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 39
  store float %tmp280, float addrspace(1)* %tmp487, align 4
  %tmp488 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 40
  store float %tmp287, float addrspace(1)* %tmp488, align 4
  %tmp489 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 41
  store float %tmp294, float addrspace(1)* %tmp489, align 4
  %tmp490 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 42
  store float %tmp301, float addrspace(1)* %tmp490, align 4
  %tmp491 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 43
  store float %tmp308, float addrspace(1)* %tmp491, align 4
  %tmp492 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 44
  store float %tmp315, float addrspace(1)* %tmp492, align 4
  %tmp493 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 45
  store float %tmp322, float addrspace(1)* %tmp493, align 4
  %tmp494 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 46
  store float %tmp329, float addrspace(1)* %tmp494, align 4
  %tmp495 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 47
  store float %tmp336, float addrspace(1)* %tmp495, align 4
  %tmp496 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 48
  store float %tmp343, float addrspace(1)* %tmp496, align 4
  %tmp497 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 49
  store float %tmp350, float addrspace(1)* %tmp497, align 4
  %tmp498 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 50
  store float %tmp357, float addrspace(1)* %tmp498, align 4
  %tmp499 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 51
  store float %tmp364, float addrspace(1)* %tmp499, align 4
  %tmp500 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 52
  store float %tmp371, float addrspace(1)* %tmp500, align 4
  %tmp501 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 53
  store float %tmp378, float addrspace(1)* %tmp501, align 4
  %tmp502 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 54
  store float %tmp385, float addrspace(1)* %tmp502, align 4
  %tmp503 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 55
  store float %tmp392, float addrspace(1)* %tmp503, align 4
  %tmp504 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 56
  store float %tmp399, float addrspace(1)* %tmp504, align 4
  %tmp505 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 57
  store float %tmp406, float addrspace(1)* %tmp505, align 4
  %tmp506 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 58
  store float %tmp413, float addrspace(1)* %tmp506, align 4
  %tmp507 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 59
  store float %tmp420, float addrspace(1)* %tmp507, align 4
  %tmp508 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 60
  store float %tmp427, float addrspace(1)* %tmp508, align 4
  %tmp509 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 61
  store float %tmp434, float addrspace(1)* %tmp509, align 4
  %tmp510 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 62
  store float %tmp441, float addrspace(1)* %tmp510, align 4
  %tmp511 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 63
  store float %tmp448, float addrspace(1)* %tmp511, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.fmuladd.f32(float, float, float) #0

attributes #0 = { nounwind readnone }
