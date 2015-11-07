; RUN: llc -march=amdgcn -mcpu=tahiti -mattr=+vgpr-spilling < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=+vgpr-spilling < %s | FileCheck %s

; FIXME: Enable -verify-instructions

; This ends up using all 255 registers and requires register
; scavenging which will fail to find an unsued register.

; Check the ScratchSize to avoid regressions from spilling
; intermediate register class copies.

; FIXME: The same register is initialized to 0 for every spill.

; CHECK-LABEL: {{^}}main:
; CHECK: NumVgprs: 256
; CHECK: ScratchSize: 1024

define void @main([9 x <16 x i8>] addrspace(2)* byval %arg, [17 x <16 x i8>] addrspace(2)* byval %arg1, [17 x <4 x i32>] addrspace(2)* byval %arg2, [34 x <8 x i32>] addrspace(2)* byval %arg3, [16 x <16 x i8>] addrspace(2)* byval %arg4, i32 inreg %arg5, i32 inreg %arg6, i32 %arg7, i32 %arg8, i32 %arg9, i32 %arg10) #0 {
bb:
  %tmp = getelementptr [17 x <16 x i8>], [17 x <16 x i8>] addrspace(2)* %arg1, i64 0, i64 0
  %tmp11 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, align 16, !tbaa !0
  %tmp12 = call float @llvm.SI.load.const(<16 x i8> %tmp11, i32 0)
  %tmp13 = call float @llvm.SI.load.const(<16 x i8> %tmp11, i32 16)
  %tmp14 = call float @llvm.SI.load.const(<16 x i8> %tmp11, i32 32)
  %tmp15 = getelementptr [16 x <16 x i8>], [16 x <16 x i8>] addrspace(2)* %arg4, i64 0, i64 0
  %tmp16 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp15, align 16, !tbaa !0
  %tmp17 = add i32 %arg5, %arg7
  %tmp18 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %tmp16, i32 0, i32 %tmp17)
  %tmp19 = extractelement <4 x float> %tmp18, i32 0
  %tmp20 = extractelement <4 x float> %tmp18, i32 1
  %tmp21 = extractelement <4 x float> %tmp18, i32 2
  %tmp22 = extractelement <4 x float> %tmp18, i32 3
  %tmp23 = bitcast float %tmp14 to i32
  br label %bb24

bb24:                                             ; preds = %bb157, %bb
  %tmp25 = phi float [ 0.000000e+00, %bb ], [ %tmp350, %bb157 ]
  %tmp26 = phi float [ 0.000000e+00, %bb ], [ %tmp349, %bb157 ]
  %tmp27 = phi float [ 0.000000e+00, %bb ], [ %tmp348, %bb157 ]
  %tmp28 = phi float [ 0.000000e+00, %bb ], [ %tmp351, %bb157 ]
  %tmp29 = phi float [ 0.000000e+00, %bb ], [ %tmp347, %bb157 ]
  %tmp30 = phi float [ 0.000000e+00, %bb ], [ %tmp346, %bb157 ]
  %tmp31 = phi float [ 0.000000e+00, %bb ], [ %tmp345, %bb157 ]
  %tmp32 = phi float [ 0.000000e+00, %bb ], [ %tmp352, %bb157 ]
  %tmp33 = phi float [ 0.000000e+00, %bb ], [ %tmp344, %bb157 ]
  %tmp34 = phi float [ 0.000000e+00, %bb ], [ %tmp343, %bb157 ]
  %tmp35 = phi float [ 0.000000e+00, %bb ], [ %tmp342, %bb157 ]
  %tmp36 = phi float [ 0.000000e+00, %bb ], [ %tmp353, %bb157 ]
  %tmp37 = phi float [ 0.000000e+00, %bb ], [ %tmp341, %bb157 ]
  %tmp38 = phi float [ 0.000000e+00, %bb ], [ %tmp340, %bb157 ]
  %tmp39 = phi float [ 0.000000e+00, %bb ], [ %tmp339, %bb157 ]
  %tmp40 = phi float [ 0.000000e+00, %bb ], [ %tmp354, %bb157 ]
  %tmp41 = phi float [ 0.000000e+00, %bb ], [ %tmp338, %bb157 ]
  %tmp42 = phi float [ 0.000000e+00, %bb ], [ %tmp337, %bb157 ]
  %tmp43 = phi float [ 0.000000e+00, %bb ], [ %tmp336, %bb157 ]
  %tmp44 = phi float [ 0.000000e+00, %bb ], [ %tmp355, %bb157 ]
  %tmp45 = phi float [ 0.000000e+00, %bb ], [ %tmp335, %bb157 ]
  %tmp46 = phi float [ 0.000000e+00, %bb ], [ %tmp334, %bb157 ]
  %tmp47 = phi float [ 0.000000e+00, %bb ], [ %tmp333, %bb157 ]
  %tmp48 = phi float [ 0.000000e+00, %bb ], [ %tmp356, %bb157 ]
  %tmp49 = phi float [ 0.000000e+00, %bb ], [ %tmp332, %bb157 ]
  %tmp50 = phi float [ 0.000000e+00, %bb ], [ %tmp331, %bb157 ]
  %tmp51 = phi float [ 0.000000e+00, %bb ], [ %tmp330, %bb157 ]
  %tmp52 = phi float [ 0.000000e+00, %bb ], [ %tmp357, %bb157 ]
  %tmp53 = phi float [ 0.000000e+00, %bb ], [ %tmp329, %bb157 ]
  %tmp54 = phi float [ 0.000000e+00, %bb ], [ %tmp328, %bb157 ]
  %tmp55 = phi float [ 0.000000e+00, %bb ], [ %tmp327, %bb157 ]
  %tmp56 = phi float [ 0.000000e+00, %bb ], [ %tmp358, %bb157 ]
  %tmp57 = phi float [ 0.000000e+00, %bb ], [ %tmp326, %bb157 ]
  %tmp58 = phi float [ 0.000000e+00, %bb ], [ %tmp325, %bb157 ]
  %tmp59 = phi float [ 0.000000e+00, %bb ], [ %tmp324, %bb157 ]
  %tmp60 = phi float [ 0.000000e+00, %bb ], [ %tmp359, %bb157 ]
  %tmp61 = phi float [ 0.000000e+00, %bb ], [ %tmp323, %bb157 ]
  %tmp62 = phi float [ 0.000000e+00, %bb ], [ %tmp322, %bb157 ]
  %tmp63 = phi float [ 0.000000e+00, %bb ], [ %tmp321, %bb157 ]
  %tmp64 = phi float [ 0.000000e+00, %bb ], [ %tmp360, %bb157 ]
  %tmp65 = phi float [ 0.000000e+00, %bb ], [ %tmp320, %bb157 ]
  %tmp66 = phi float [ 0.000000e+00, %bb ], [ %tmp319, %bb157 ]
  %tmp67 = phi float [ 0.000000e+00, %bb ], [ %tmp318, %bb157 ]
  %tmp68 = phi float [ 0.000000e+00, %bb ], [ %tmp361, %bb157 ]
  %tmp69 = phi float [ 0.000000e+00, %bb ], [ %tmp317, %bb157 ]
  %tmp70 = phi float [ 0.000000e+00, %bb ], [ %tmp316, %bb157 ]
  %tmp71 = phi float [ 0.000000e+00, %bb ], [ %tmp315, %bb157 ]
  %tmp72 = phi float [ 0.000000e+00, %bb ], [ %tmp362, %bb157 ]
  %tmp73 = phi float [ 0.000000e+00, %bb ], [ %tmp314, %bb157 ]
  %tmp74 = phi float [ 0.000000e+00, %bb ], [ %tmp313, %bb157 ]
  %tmp75 = phi float [ 0.000000e+00, %bb ], [ %tmp312, %bb157 ]
  %tmp76 = phi float [ 0.000000e+00, %bb ], [ %tmp363, %bb157 ]
  %tmp77 = phi float [ 0.000000e+00, %bb ], [ %tmp311, %bb157 ]
  %tmp78 = phi float [ 0.000000e+00, %bb ], [ %tmp310, %bb157 ]
  %tmp79 = phi float [ 0.000000e+00, %bb ], [ %tmp309, %bb157 ]
  %tmp80 = phi float [ 0.000000e+00, %bb ], [ %tmp364, %bb157 ]
  %tmp81 = phi float [ 0.000000e+00, %bb ], [ %tmp308, %bb157 ]
  %tmp82 = phi float [ 0.000000e+00, %bb ], [ %tmp307, %bb157 ]
  %tmp83 = phi float [ 0.000000e+00, %bb ], [ %tmp306, %bb157 ]
  %tmp84 = phi float [ 0.000000e+00, %bb ], [ %tmp365, %bb157 ]
  %tmp85 = phi float [ 0.000000e+00, %bb ], [ %tmp305, %bb157 ]
  %tmp86 = phi float [ 0.000000e+00, %bb ], [ %tmp304, %bb157 ]
  %tmp87 = phi float [ 0.000000e+00, %bb ], [ %tmp303, %bb157 ]
  %tmp88 = phi float [ 0.000000e+00, %bb ], [ %tmp366, %bb157 ]
  %tmp89 = phi float [ 0.000000e+00, %bb ], [ %tmp302, %bb157 ]
  %tmp90 = phi float [ 0.000000e+00, %bb ], [ %tmp301, %bb157 ]
  %tmp91 = phi float [ 0.000000e+00, %bb ], [ %tmp300, %bb157 ]
  %tmp92 = phi float [ 0.000000e+00, %bb ], [ %tmp367, %bb157 ]
  %tmp93 = phi float [ 0.000000e+00, %bb ], [ %tmp299, %bb157 ]
  %tmp94 = phi float [ 0.000000e+00, %bb ], [ %tmp298, %bb157 ]
  %tmp95 = phi float [ 0.000000e+00, %bb ], [ %tmp297, %bb157 ]
  %tmp96 = phi float [ 0.000000e+00, %bb ], [ %tmp368, %bb157 ]
  %tmp97 = phi float [ 0.000000e+00, %bb ], [ %tmp296, %bb157 ]
  %tmp98 = phi float [ 0.000000e+00, %bb ], [ %tmp295, %bb157 ]
  %tmp99 = phi float [ 0.000000e+00, %bb ], [ %tmp294, %bb157 ]
  %tmp100 = phi float [ 0.000000e+00, %bb ], [ %tmp369, %bb157 ]
  %tmp101 = phi float [ 0.000000e+00, %bb ], [ %tmp293, %bb157 ]
  %tmp102 = phi float [ 0.000000e+00, %bb ], [ %tmp292, %bb157 ]
  %tmp103 = phi float [ 0.000000e+00, %bb ], [ %tmp291, %bb157 ]
  %tmp104 = phi float [ 0.000000e+00, %bb ], [ %tmp370, %bb157 ]
  %tmp105 = phi float [ 0.000000e+00, %bb ], [ %tmp371, %bb157 ]
  %tmp106 = phi float [ 0.000000e+00, %bb ], [ %tmp372, %bb157 ]
  %tmp107 = phi float [ 0.000000e+00, %bb ], [ %tmp421, %bb157 ]
  %tmp108 = phi float [ 0.000000e+00, %bb ], [ %tmp373, %bb157 ]
  %tmp109 = phi float [ 0.000000e+00, %bb ], [ %tmp374, %bb157 ]
  %tmp110 = phi float [ 0.000000e+00, %bb ], [ %tmp375, %bb157 ]
  %tmp111 = phi float [ 0.000000e+00, %bb ], [ %tmp376, %bb157 ]
  %tmp112 = phi float [ 0.000000e+00, %bb ], [ %tmp377, %bb157 ]
  %tmp113 = phi float [ 0.000000e+00, %bb ], [ %tmp378, %bb157 ]
  %tmp114 = phi float [ 0.000000e+00, %bb ], [ %tmp379, %bb157 ]
  %tmp115 = phi float [ 0.000000e+00, %bb ], [ %tmp380, %bb157 ]
  %tmp116 = phi float [ 0.000000e+00, %bb ], [ %tmp381, %bb157 ]
  %tmp117 = phi float [ 0.000000e+00, %bb ], [ %tmp382, %bb157 ]
  %tmp118 = phi float [ 0.000000e+00, %bb ], [ %tmp383, %bb157 ]
  %tmp119 = phi float [ 0.000000e+00, %bb ], [ %tmp384, %bb157 ]
  %tmp120 = phi float [ 0.000000e+00, %bb ], [ %tmp385, %bb157 ]
  %tmp121 = phi float [ 0.000000e+00, %bb ], [ %tmp386, %bb157 ]
  %tmp122 = phi float [ 0.000000e+00, %bb ], [ %tmp387, %bb157 ]
  %tmp123 = phi float [ 0.000000e+00, %bb ], [ %tmp388, %bb157 ]
  %tmp124 = phi float [ 0.000000e+00, %bb ], [ %tmp389, %bb157 ]
  %tmp125 = phi float [ 0.000000e+00, %bb ], [ %tmp390, %bb157 ]
  %tmp126 = phi float [ 0.000000e+00, %bb ], [ %tmp391, %bb157 ]
  %tmp127 = phi float [ 0.000000e+00, %bb ], [ %tmp392, %bb157 ]
  %tmp128 = phi float [ 0.000000e+00, %bb ], [ %tmp393, %bb157 ]
  %tmp129 = phi float [ 0.000000e+00, %bb ], [ %tmp394, %bb157 ]
  %tmp130 = phi float [ 0.000000e+00, %bb ], [ %tmp395, %bb157 ]
  %tmp131 = phi float [ 0.000000e+00, %bb ], [ %tmp396, %bb157 ]
  %tmp132 = phi float [ 0.000000e+00, %bb ], [ %tmp397, %bb157 ]
  %tmp133 = phi float [ 0.000000e+00, %bb ], [ %tmp398, %bb157 ]
  %tmp134 = phi float [ 0.000000e+00, %bb ], [ %tmp399, %bb157 ]
  %tmp135 = phi float [ 0.000000e+00, %bb ], [ %tmp400, %bb157 ]
  %tmp136 = phi float [ 0.000000e+00, %bb ], [ %tmp401, %bb157 ]
  %tmp137 = phi float [ 0.000000e+00, %bb ], [ %tmp402, %bb157 ]
  %tmp138 = phi float [ 0.000000e+00, %bb ], [ %tmp403, %bb157 ]
  %tmp139 = phi float [ 0.000000e+00, %bb ], [ %tmp404, %bb157 ]
  %tmp140 = phi float [ 0.000000e+00, %bb ], [ %tmp405, %bb157 ]
  %tmp141 = phi float [ 0.000000e+00, %bb ], [ %tmp406, %bb157 ]
  %tmp142 = phi float [ 0.000000e+00, %bb ], [ %tmp407, %bb157 ]
  %tmp143 = phi float [ 0.000000e+00, %bb ], [ %tmp408, %bb157 ]
  %tmp144 = phi float [ 0.000000e+00, %bb ], [ %tmp409, %bb157 ]
  %tmp145 = phi float [ 0.000000e+00, %bb ], [ %tmp410, %bb157 ]
  %tmp146 = phi float [ 0.000000e+00, %bb ], [ %tmp411, %bb157 ]
  %tmp147 = phi float [ 0.000000e+00, %bb ], [ %tmp412, %bb157 ]
  %tmp148 = phi float [ 0.000000e+00, %bb ], [ %tmp413, %bb157 ]
  %tmp149 = phi float [ 0.000000e+00, %bb ], [ %tmp414, %bb157 ]
  %tmp150 = phi float [ 0.000000e+00, %bb ], [ %tmp415, %bb157 ]
  %tmp151 = phi float [ 0.000000e+00, %bb ], [ %tmp416, %bb157 ]
  %tmp152 = phi float [ 0.000000e+00, %bb ], [ %tmp417, %bb157 ]
  %tmp153 = phi float [ 0.000000e+00, %bb ], [ %tmp418, %bb157 ]
  %tmp154 = bitcast float %tmp107 to i32
  %tmp155 = icmp sgt i32 %tmp154, 125
  br i1 %tmp155, label %bb156, label %bb157

bb156:                                            ; preds = %bb24
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %tmp12, float %tmp103, float %tmp102, float %tmp101)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 33, i32 0, float %tmp99, float %tmp98, float %tmp97, float %tmp95)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 34, i32 0, float %tmp94, float %tmp93, float %tmp91, float %tmp90)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 35, i32 0, float %tmp89, float %tmp87, float %tmp86, float %tmp85)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 36, i32 0, float %tmp83, float %tmp82, float %tmp81, float %tmp79)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 37, i32 0, float %tmp78, float %tmp77, float %tmp75, float %tmp74)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 38, i32 0, float %tmp73, float %tmp71, float %tmp70, float %tmp69)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 39, i32 0, float %tmp67, float %tmp66, float %tmp65, float %tmp63)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 40, i32 0, float %tmp62, float %tmp61, float %tmp59, float %tmp58)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 41, i32 0, float %tmp57, float %tmp55, float %tmp54, float %tmp53)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 42, i32 0, float %tmp51, float %tmp50, float %tmp49, float %tmp47)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 43, i32 0, float %tmp46, float %tmp45, float %tmp43, float %tmp42)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 44, i32 0, float %tmp41, float %tmp39, float %tmp38, float %tmp37)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 45, i32 0, float %tmp35, float %tmp34, float %tmp33, float %tmp31)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 46, i32 0, float %tmp30, float %tmp29, float %tmp27, float %tmp26)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 47, i32 0, float %tmp25, float %tmp28, float %tmp32, float %tmp36)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 48, i32 0, float %tmp40, float %tmp44, float %tmp48, float %tmp52)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 49, i32 0, float %tmp56, float %tmp60, float %tmp64, float %tmp68)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 50, i32 0, float %tmp72, float %tmp76, float %tmp80, float %tmp84)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 51, i32 0, float %tmp88, float %tmp92, float %tmp96, float %tmp100)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 52, i32 0, float %tmp104, float %tmp105, float %tmp106, float %tmp108)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 53, i32 0, float %tmp109, float %tmp110, float %tmp111, float %tmp112)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 54, i32 0, float %tmp113, float %tmp114, float %tmp115, float %tmp116)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 55, i32 0, float %tmp117, float %tmp118, float %tmp119, float %tmp120)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 56, i32 0, float %tmp121, float %tmp122, float %tmp123, float %tmp124)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 57, i32 0, float %tmp125, float %tmp126, float %tmp127, float %tmp128)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 58, i32 0, float %tmp129, float %tmp130, float %tmp131, float %tmp132)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 59, i32 0, float %tmp133, float %tmp134, float %tmp135, float %tmp136)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 60, i32 0, float %tmp137, float %tmp138, float %tmp139, float %tmp140)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 61, i32 0, float %tmp141, float %tmp142, float %tmp143, float %tmp144)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 62, i32 0, float %tmp145, float %tmp146, float %tmp147, float %tmp148)
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 63, i32 0, float %tmp149, float %tmp150, float %tmp151, float %tmp13)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %tmp19, float %tmp20, float %tmp21, float %tmp22)
  ret void

bb157:                                            ; preds = %bb24
  %tmp158 = bitcast float %tmp107 to i32
  %tmp159 = bitcast float %tmp107 to i32
  %tmp160 = add i32 %tmp23, %tmp159
  %tmp161 = bitcast i32 %tmp160 to float
  %tmp162 = insertelement <128 x float> undef, float %tmp103, i32 0
  %tmp163 = insertelement <128 x float> %tmp162, float %tmp102, i32 1
  %tmp164 = insertelement <128 x float> %tmp163, float %tmp101, i32 2
  %tmp165 = insertelement <128 x float> %tmp164, float %tmp99, i32 3
  %tmp166 = insertelement <128 x float> %tmp165, float %tmp98, i32 4
  %tmp167 = insertelement <128 x float> %tmp166, float %tmp97, i32 5
  %tmp168 = insertelement <128 x float> %tmp167, float %tmp95, i32 6
  %tmp169 = insertelement <128 x float> %tmp168, float %tmp94, i32 7
  %tmp170 = insertelement <128 x float> %tmp169, float %tmp93, i32 8
  %tmp171 = insertelement <128 x float> %tmp170, float %tmp91, i32 9
  %tmp172 = insertelement <128 x float> %tmp171, float %tmp90, i32 10
  %tmp173 = insertelement <128 x float> %tmp172, float %tmp89, i32 11
  %tmp174 = insertelement <128 x float> %tmp173, float %tmp87, i32 12
  %tmp175 = insertelement <128 x float> %tmp174, float %tmp86, i32 13
  %tmp176 = insertelement <128 x float> %tmp175, float %tmp85, i32 14
  %tmp177 = insertelement <128 x float> %tmp176, float %tmp83, i32 15
  %tmp178 = insertelement <128 x float> %tmp177, float %tmp82, i32 16
  %tmp179 = insertelement <128 x float> %tmp178, float %tmp81, i32 17
  %tmp180 = insertelement <128 x float> %tmp179, float %tmp79, i32 18
  %tmp181 = insertelement <128 x float> %tmp180, float %tmp78, i32 19
  %tmp182 = insertelement <128 x float> %tmp181, float %tmp77, i32 20
  %tmp183 = insertelement <128 x float> %tmp182, float %tmp75, i32 21
  %tmp184 = insertelement <128 x float> %tmp183, float %tmp74, i32 22
  %tmp185 = insertelement <128 x float> %tmp184, float %tmp73, i32 23
  %tmp186 = insertelement <128 x float> %tmp185, float %tmp71, i32 24
  %tmp187 = insertelement <128 x float> %tmp186, float %tmp70, i32 25
  %tmp188 = insertelement <128 x float> %tmp187, float %tmp69, i32 26
  %tmp189 = insertelement <128 x float> %tmp188, float %tmp67, i32 27
  %tmp190 = insertelement <128 x float> %tmp189, float %tmp66, i32 28
  %tmp191 = insertelement <128 x float> %tmp190, float %tmp65, i32 29
  %tmp192 = insertelement <128 x float> %tmp191, float %tmp63, i32 30
  %tmp193 = insertelement <128 x float> %tmp192, float %tmp62, i32 31
  %tmp194 = insertelement <128 x float> %tmp193, float %tmp61, i32 32
  %tmp195 = insertelement <128 x float> %tmp194, float %tmp59, i32 33
  %tmp196 = insertelement <128 x float> %tmp195, float %tmp58, i32 34
  %tmp197 = insertelement <128 x float> %tmp196, float %tmp57, i32 35
  %tmp198 = insertelement <128 x float> %tmp197, float %tmp55, i32 36
  %tmp199 = insertelement <128 x float> %tmp198, float %tmp54, i32 37
  %tmp200 = insertelement <128 x float> %tmp199, float %tmp53, i32 38
  %tmp201 = insertelement <128 x float> %tmp200, float %tmp51, i32 39
  %tmp202 = insertelement <128 x float> %tmp201, float %tmp50, i32 40
  %tmp203 = insertelement <128 x float> %tmp202, float %tmp49, i32 41
  %tmp204 = insertelement <128 x float> %tmp203, float %tmp47, i32 42
  %tmp205 = insertelement <128 x float> %tmp204, float %tmp46, i32 43
  %tmp206 = insertelement <128 x float> %tmp205, float %tmp45, i32 44
  %tmp207 = insertelement <128 x float> %tmp206, float %tmp43, i32 45
  %tmp208 = insertelement <128 x float> %tmp207, float %tmp42, i32 46
  %tmp209 = insertelement <128 x float> %tmp208, float %tmp41, i32 47
  %tmp210 = insertelement <128 x float> %tmp209, float %tmp39, i32 48
  %tmp211 = insertelement <128 x float> %tmp210, float %tmp38, i32 49
  %tmp212 = insertelement <128 x float> %tmp211, float %tmp37, i32 50
  %tmp213 = insertelement <128 x float> %tmp212, float %tmp35, i32 51
  %tmp214 = insertelement <128 x float> %tmp213, float %tmp34, i32 52
  %tmp215 = insertelement <128 x float> %tmp214, float %tmp33, i32 53
  %tmp216 = insertelement <128 x float> %tmp215, float %tmp31, i32 54
  %tmp217 = insertelement <128 x float> %tmp216, float %tmp30, i32 55
  %tmp218 = insertelement <128 x float> %tmp217, float %tmp29, i32 56
  %tmp219 = insertelement <128 x float> %tmp218, float %tmp27, i32 57
  %tmp220 = insertelement <128 x float> %tmp219, float %tmp26, i32 58
  %tmp221 = insertelement <128 x float> %tmp220, float %tmp25, i32 59
  %tmp222 = insertelement <128 x float> %tmp221, float %tmp28, i32 60
  %tmp223 = insertelement <128 x float> %tmp222, float %tmp32, i32 61
  %tmp224 = insertelement <128 x float> %tmp223, float %tmp36, i32 62
  %tmp225 = insertelement <128 x float> %tmp224, float %tmp40, i32 63
  %tmp226 = insertelement <128 x float> %tmp225, float %tmp44, i32 64
  %tmp227 = insertelement <128 x float> %tmp226, float %tmp48, i32 65
  %tmp228 = insertelement <128 x float> %tmp227, float %tmp52, i32 66
  %tmp229 = insertelement <128 x float> %tmp228, float %tmp56, i32 67
  %tmp230 = insertelement <128 x float> %tmp229, float %tmp60, i32 68
  %tmp231 = insertelement <128 x float> %tmp230, float %tmp64, i32 69
  %tmp232 = insertelement <128 x float> %tmp231, float %tmp68, i32 70
  %tmp233 = insertelement <128 x float> %tmp232, float %tmp72, i32 71
  %tmp234 = insertelement <128 x float> %tmp233, float %tmp76, i32 72
  %tmp235 = insertelement <128 x float> %tmp234, float %tmp80, i32 73
  %tmp236 = insertelement <128 x float> %tmp235, float %tmp84, i32 74
  %tmp237 = insertelement <128 x float> %tmp236, float %tmp88, i32 75
  %tmp238 = insertelement <128 x float> %tmp237, float %tmp92, i32 76
  %tmp239 = insertelement <128 x float> %tmp238, float %tmp96, i32 77
  %tmp240 = insertelement <128 x float> %tmp239, float %tmp100, i32 78
  %tmp241 = insertelement <128 x float> %tmp240, float %tmp104, i32 79
  %tmp242 = insertelement <128 x float> %tmp241, float %tmp105, i32 80
  %tmp243 = insertelement <128 x float> %tmp242, float %tmp106, i32 81
  %tmp244 = insertelement <128 x float> %tmp243, float %tmp108, i32 82
  %tmp245 = insertelement <128 x float> %tmp244, float %tmp109, i32 83
  %tmp246 = insertelement <128 x float> %tmp245, float %tmp110, i32 84
  %tmp247 = insertelement <128 x float> %tmp246, float %tmp111, i32 85
  %tmp248 = insertelement <128 x float> %tmp247, float %tmp112, i32 86
  %tmp249 = insertelement <128 x float> %tmp248, float %tmp113, i32 87
  %tmp250 = insertelement <128 x float> %tmp249, float %tmp114, i32 88
  %tmp251 = insertelement <128 x float> %tmp250, float %tmp115, i32 89
  %tmp252 = insertelement <128 x float> %tmp251, float %tmp116, i32 90
  %tmp253 = insertelement <128 x float> %tmp252, float %tmp117, i32 91
  %tmp254 = insertelement <128 x float> %tmp253, float %tmp118, i32 92
  %tmp255 = insertelement <128 x float> %tmp254, float %tmp119, i32 93
  %tmp256 = insertelement <128 x float> %tmp255, float %tmp120, i32 94
  %tmp257 = insertelement <128 x float> %tmp256, float %tmp121, i32 95
  %tmp258 = insertelement <128 x float> %tmp257, float %tmp122, i32 96
  %tmp259 = insertelement <128 x float> %tmp258, float %tmp123, i32 97
  %tmp260 = insertelement <128 x float> %tmp259, float %tmp124, i32 98
  %tmp261 = insertelement <128 x float> %tmp260, float %tmp125, i32 99
  %tmp262 = insertelement <128 x float> %tmp261, float %tmp126, i32 100
  %tmp263 = insertelement <128 x float> %tmp262, float %tmp127, i32 101
  %tmp264 = insertelement <128 x float> %tmp263, float %tmp128, i32 102
  %tmp265 = insertelement <128 x float> %tmp264, float %tmp129, i32 103
  %tmp266 = insertelement <128 x float> %tmp265, float %tmp130, i32 104
  %tmp267 = insertelement <128 x float> %tmp266, float %tmp131, i32 105
  %tmp268 = insertelement <128 x float> %tmp267, float %tmp132, i32 106
  %tmp269 = insertelement <128 x float> %tmp268, float %tmp133, i32 107
  %tmp270 = insertelement <128 x float> %tmp269, float %tmp134, i32 108
  %tmp271 = insertelement <128 x float> %tmp270, float %tmp135, i32 109
  %tmp272 = insertelement <128 x float> %tmp271, float %tmp136, i32 110
  %tmp273 = insertelement <128 x float> %tmp272, float %tmp137, i32 111
  %tmp274 = insertelement <128 x float> %tmp273, float %tmp138, i32 112
  %tmp275 = insertelement <128 x float> %tmp274, float %tmp139, i32 113
  %tmp276 = insertelement <128 x float> %tmp275, float %tmp140, i32 114
  %tmp277 = insertelement <128 x float> %tmp276, float %tmp141, i32 115
  %tmp278 = insertelement <128 x float> %tmp277, float %tmp142, i32 116
  %tmp279 = insertelement <128 x float> %tmp278, float %tmp143, i32 117
  %tmp280 = insertelement <128 x float> %tmp279, float %tmp144, i32 118
  %tmp281 = insertelement <128 x float> %tmp280, float %tmp145, i32 119
  %tmp282 = insertelement <128 x float> %tmp281, float %tmp146, i32 120
  %tmp283 = insertelement <128 x float> %tmp282, float %tmp147, i32 121
  %tmp284 = insertelement <128 x float> %tmp283, float %tmp148, i32 122
  %tmp285 = insertelement <128 x float> %tmp284, float %tmp149, i32 123
  %tmp286 = insertelement <128 x float> %tmp285, float %tmp150, i32 124
  %tmp287 = insertelement <128 x float> %tmp286, float %tmp151, i32 125
  %tmp288 = insertelement <128 x float> %tmp287, float %tmp152, i32 126
  %tmp289 = insertelement <128 x float> %tmp288, float %tmp153, i32 127
  %tmp290 = insertelement <128 x float> %tmp289, float %tmp161, i32 %tmp158
  %tmp291 = extractelement <128 x float> %tmp290, i32 0
  %tmp292 = extractelement <128 x float> %tmp290, i32 1
  %tmp293 = extractelement <128 x float> %tmp290, i32 2
  %tmp294 = extractelement <128 x float> %tmp290, i32 3
  %tmp295 = extractelement <128 x float> %tmp290, i32 4
  %tmp296 = extractelement <128 x float> %tmp290, i32 5
  %tmp297 = extractelement <128 x float> %tmp290, i32 6
  %tmp298 = extractelement <128 x float> %tmp290, i32 7
  %tmp299 = extractelement <128 x float> %tmp290, i32 8
  %tmp300 = extractelement <128 x float> %tmp290, i32 9
  %tmp301 = extractelement <128 x float> %tmp290, i32 10
  %tmp302 = extractelement <128 x float> %tmp290, i32 11
  %tmp303 = extractelement <128 x float> %tmp290, i32 12
  %tmp304 = extractelement <128 x float> %tmp290, i32 13
  %tmp305 = extractelement <128 x float> %tmp290, i32 14
  %tmp306 = extractelement <128 x float> %tmp290, i32 15
  %tmp307 = extractelement <128 x float> %tmp290, i32 16
  %tmp308 = extractelement <128 x float> %tmp290, i32 17
  %tmp309 = extractelement <128 x float> %tmp290, i32 18
  %tmp310 = extractelement <128 x float> %tmp290, i32 19
  %tmp311 = extractelement <128 x float> %tmp290, i32 20
  %tmp312 = extractelement <128 x float> %tmp290, i32 21
  %tmp313 = extractelement <128 x float> %tmp290, i32 22
  %tmp314 = extractelement <128 x float> %tmp290, i32 23
  %tmp315 = extractelement <128 x float> %tmp290, i32 24
  %tmp316 = extractelement <128 x float> %tmp290, i32 25
  %tmp317 = extractelement <128 x float> %tmp290, i32 26
  %tmp318 = extractelement <128 x float> %tmp290, i32 27
  %tmp319 = extractelement <128 x float> %tmp290, i32 28
  %tmp320 = extractelement <128 x float> %tmp290, i32 29
  %tmp321 = extractelement <128 x float> %tmp290, i32 30
  %tmp322 = extractelement <128 x float> %tmp290, i32 31
  %tmp323 = extractelement <128 x float> %tmp290, i32 32
  %tmp324 = extractelement <128 x float> %tmp290, i32 33
  %tmp325 = extractelement <128 x float> %tmp290, i32 34
  %tmp326 = extractelement <128 x float> %tmp290, i32 35
  %tmp327 = extractelement <128 x float> %tmp290, i32 36
  %tmp328 = extractelement <128 x float> %tmp290, i32 37
  %tmp329 = extractelement <128 x float> %tmp290, i32 38
  %tmp330 = extractelement <128 x float> %tmp290, i32 39
  %tmp331 = extractelement <128 x float> %tmp290, i32 40
  %tmp332 = extractelement <128 x float> %tmp290, i32 41
  %tmp333 = extractelement <128 x float> %tmp290, i32 42
  %tmp334 = extractelement <128 x float> %tmp290, i32 43
  %tmp335 = extractelement <128 x float> %tmp290, i32 44
  %tmp336 = extractelement <128 x float> %tmp290, i32 45
  %tmp337 = extractelement <128 x float> %tmp290, i32 46
  %tmp338 = extractelement <128 x float> %tmp290, i32 47
  %tmp339 = extractelement <128 x float> %tmp290, i32 48
  %tmp340 = extractelement <128 x float> %tmp290, i32 49
  %tmp341 = extractelement <128 x float> %tmp290, i32 50
  %tmp342 = extractelement <128 x float> %tmp290, i32 51
  %tmp343 = extractelement <128 x float> %tmp290, i32 52
  %tmp344 = extractelement <128 x float> %tmp290, i32 53
  %tmp345 = extractelement <128 x float> %tmp290, i32 54
  %tmp346 = extractelement <128 x float> %tmp290, i32 55
  %tmp347 = extractelement <128 x float> %tmp290, i32 56
  %tmp348 = extractelement <128 x float> %tmp290, i32 57
  %tmp349 = extractelement <128 x float> %tmp290, i32 58
  %tmp350 = extractelement <128 x float> %tmp290, i32 59
  %tmp351 = extractelement <128 x float> %tmp290, i32 60
  %tmp352 = extractelement <128 x float> %tmp290, i32 61
  %tmp353 = extractelement <128 x float> %tmp290, i32 62
  %tmp354 = extractelement <128 x float> %tmp290, i32 63
  %tmp355 = extractelement <128 x float> %tmp290, i32 64
  %tmp356 = extractelement <128 x float> %tmp290, i32 65
  %tmp357 = extractelement <128 x float> %tmp290, i32 66
  %tmp358 = extractelement <128 x float> %tmp290, i32 67
  %tmp359 = extractelement <128 x float> %tmp290, i32 68
  %tmp360 = extractelement <128 x float> %tmp290, i32 69
  %tmp361 = extractelement <128 x float> %tmp290, i32 70
  %tmp362 = extractelement <128 x float> %tmp290, i32 71
  %tmp363 = extractelement <128 x float> %tmp290, i32 72
  %tmp364 = extractelement <128 x float> %tmp290, i32 73
  %tmp365 = extractelement <128 x float> %tmp290, i32 74
  %tmp366 = extractelement <128 x float> %tmp290, i32 75
  %tmp367 = extractelement <128 x float> %tmp290, i32 76
  %tmp368 = extractelement <128 x float> %tmp290, i32 77
  %tmp369 = extractelement <128 x float> %tmp290, i32 78
  %tmp370 = extractelement <128 x float> %tmp290, i32 79
  %tmp371 = extractelement <128 x float> %tmp290, i32 80
  %tmp372 = extractelement <128 x float> %tmp290, i32 81
  %tmp373 = extractelement <128 x float> %tmp290, i32 82
  %tmp374 = extractelement <128 x float> %tmp290, i32 83
  %tmp375 = extractelement <128 x float> %tmp290, i32 84
  %tmp376 = extractelement <128 x float> %tmp290, i32 85
  %tmp377 = extractelement <128 x float> %tmp290, i32 86
  %tmp378 = extractelement <128 x float> %tmp290, i32 87
  %tmp379 = extractelement <128 x float> %tmp290, i32 88
  %tmp380 = extractelement <128 x float> %tmp290, i32 89
  %tmp381 = extractelement <128 x float> %tmp290, i32 90
  %tmp382 = extractelement <128 x float> %tmp290, i32 91
  %tmp383 = extractelement <128 x float> %tmp290, i32 92
  %tmp384 = extractelement <128 x float> %tmp290, i32 93
  %tmp385 = extractelement <128 x float> %tmp290, i32 94
  %tmp386 = extractelement <128 x float> %tmp290, i32 95
  %tmp387 = extractelement <128 x float> %tmp290, i32 96
  %tmp388 = extractelement <128 x float> %tmp290, i32 97
  %tmp389 = extractelement <128 x float> %tmp290, i32 98
  %tmp390 = extractelement <128 x float> %tmp290, i32 99
  %tmp391 = extractelement <128 x float> %tmp290, i32 100
  %tmp392 = extractelement <128 x float> %tmp290, i32 101
  %tmp393 = extractelement <128 x float> %tmp290, i32 102
  %tmp394 = extractelement <128 x float> %tmp290, i32 103
  %tmp395 = extractelement <128 x float> %tmp290, i32 104
  %tmp396 = extractelement <128 x float> %tmp290, i32 105
  %tmp397 = extractelement <128 x float> %tmp290, i32 106
  %tmp398 = extractelement <128 x float> %tmp290, i32 107
  %tmp399 = extractelement <128 x float> %tmp290, i32 108
  %tmp400 = extractelement <128 x float> %tmp290, i32 109
  %tmp401 = extractelement <128 x float> %tmp290, i32 110
  %tmp402 = extractelement <128 x float> %tmp290, i32 111
  %tmp403 = extractelement <128 x float> %tmp290, i32 112
  %tmp404 = extractelement <128 x float> %tmp290, i32 113
  %tmp405 = extractelement <128 x float> %tmp290, i32 114
  %tmp406 = extractelement <128 x float> %tmp290, i32 115
  %tmp407 = extractelement <128 x float> %tmp290, i32 116
  %tmp408 = extractelement <128 x float> %tmp290, i32 117
  %tmp409 = extractelement <128 x float> %tmp290, i32 118
  %tmp410 = extractelement <128 x float> %tmp290, i32 119
  %tmp411 = extractelement <128 x float> %tmp290, i32 120
  %tmp412 = extractelement <128 x float> %tmp290, i32 121
  %tmp413 = extractelement <128 x float> %tmp290, i32 122
  %tmp414 = extractelement <128 x float> %tmp290, i32 123
  %tmp415 = extractelement <128 x float> %tmp290, i32 124
  %tmp416 = extractelement <128 x float> %tmp290, i32 125
  %tmp417 = extractelement <128 x float> %tmp290, i32 126
  %tmp418 = extractelement <128 x float> %tmp290, i32 127
  %tmp419 = bitcast float %tmp107 to i32
  %tmp420 = add i32 %tmp419, 1
  %tmp421 = bitcast i32 %tmp420 to float
  br label %bb24
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.vs.load.input(<16 x i8>, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="1" "enable-no-nans-fp-math"="true" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", null}
