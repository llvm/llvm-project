; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s

%struct.ShapeData = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32, i32, i64, <4 x float>, i32, i8, i8, i16, i32, i32 }

declare float @llvm.fmuladd.f32(float, float, float)

declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)

declare ptr addrspace(1) @Scene_getSubShapeData(i32, ptr addrspace(1), ptr addrspace(1)) local_unnamed_addr

define <4 x float> @Scene_transformT(i32 %subshapeIdx, <4 x float> %v, float %time, ptr addrspace(1) %gScene, ptr addrspace(1) %gSceneOffsets) local_unnamed_addr !dbg !110 {
entry:
  ; CHECK: v_mov_b32_e32 v[[COPIED_ARG_PIECE:[0-9]+]], v9

  ; CHECK: ;DEBUG_VALUE: Scene_transformT:gScene <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr6
  ; CHECK: ;DEBUG_VALUE: Scene_transformT:gScene <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr7
  call void @llvm.dbg.value(metadata ptr addrspace(1) %gScene, metadata !120, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !154
  ; CHECK: ;DEBUG_VALUE: Scene_transformT:gSceneOffsets <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 0 32] $vgpr8
  ; CHECK: ;DEBUG_VALUE: Scene_transformT:gSceneOffsets <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef, DW_OP_LLVM_fragment 32 32] $vgpr[[COPIED_ARG_PIECE]]
  call void @llvm.dbg.value(metadata ptr addrspace(1) %gSceneOffsets, metadata !121, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)), !dbg !155
  %call = tail call ptr addrspace(1) @Scene_getSubShapeData(i32 %subshapeIdx, ptr addrspace(1) %gScene, ptr addrspace(1) %gSceneOffsets)
  %m_linearMotion = getelementptr inbounds %struct.ShapeData, ptr addrspace(1) %call, i64 0, i32 2
  %tmp = load <4 x float>, ptr addrspace(1) %m_linearMotion, align 16
  %m_angularMotion = getelementptr inbounds %struct.ShapeData, ptr addrspace(1) %call, i64 0, i32 3
  %tmp1 = load <4 x float>, ptr addrspace(1) %m_angularMotion, align 16
  %m_scaleMotion = getelementptr inbounds %struct.ShapeData, ptr addrspace(1) %call, i64 0, i32 4
  %tmp2 = load <4 x float>, ptr addrspace(1) %m_scaleMotion, align 16
  %splat.splatinsert = insertelement <4 x float> undef, float %time, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  %tmp3 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %tmp2, <4 x float> %splat.splat, <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>)
  %tmp4 = load <4 x float>, ptr addrspace(1) %call, align 16
  %m_quaternion = getelementptr inbounds %struct.ShapeData, ptr addrspace(1) %call, i64 0, i32 1
  %tmp5 = load <4 x float>, ptr addrspace(1) %m_quaternion, align 16
  %m_scale = getelementptr inbounds %struct.ShapeData, ptr addrspace(1) %call, i64 0, i32 8
  %tmp6 = load <4 x float>, ptr addrspace(1) %m_scale, align 16
  %mul = fmul <4 x float> %tmp6, %v
  %tmp7 = extractelement <4 x float> %tmp5, i64 0
  %sub.i.i = fsub float -0.000000e+00, %tmp7
  %vecinit.i.i = insertelement <4 x float> undef, float %sub.i.i, i32 0
  %tmp8 = extractelement <4 x float> %tmp5, i64 1
  %sub1.i.i = fsub float -0.000000e+00, %tmp8
  %vecinit2.i.i = insertelement <4 x float> %vecinit.i.i, float %sub1.i.i, i32 1
  %tmp9 = extractelement <4 x float> %tmp5, i64 2
  %sub3.i.i = fsub float -0.000000e+00, %tmp9
  %vecinit4.i.i = insertelement <4 x float> %vecinit2.i.i, float %sub3.i.i, i32 2
  %vecinit5.i.i = shufflevector <4 x float> %vecinit4.i.i, <4 x float> %tmp5, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %tmp10 = insertelement <4 x float> %mul, float 0.000000e+00, i64 3
  %tmp11 = extractelement <4 x float> %mul, i64 2
  %tmp12 = extractelement <4 x float> %mul, i64 1
  %tmp13 = fmul float %tmp9, %tmp12
  %tmp14 = fsub float -0.000000e+00, %tmp13
  %tmp15 = tail call float @llvm.fmuladd.f32(float %tmp8, float %tmp11, float %tmp14)
  %tmp16 = extractelement <4 x float> %mul, i64 0
  %tmp17 = fmul float %tmp7, %tmp11
  %tmp18 = fsub float -0.000000e+00, %tmp17
  %tmp19 = tail call float @llvm.fmuladd.f32(float %tmp9, float %tmp16, float %tmp18)
  %tmp20 = fmul float %tmp8, %tmp16
  %tmp21 = fsub float -0.000000e+00, %tmp20
  %tmp22 = tail call float @llvm.fmuladd.f32(float %tmp7, float %tmp12, float %tmp21)
  %tmp23 = insertelement <4 x float> <float undef, float undef, float undef, float 0.000000e+00>, float %tmp15, i32 0
  %tmp24 = insertelement <4 x float> %tmp23, float %tmp19, i32 1
  %tmp25 = insertelement <4 x float> %tmp24, float %tmp22, i32 2
  %tmp26 = extractelement <4 x float> %tmp5, i64 3
  %splat.splat.i8.i = shufflevector <4 x float> %tmp5, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %splat.splat2.i9.i = shufflevector <4 x float> %tmp10, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul3.i10.i = fmul <4 x float> %tmp5, %splat.splat2.i9.i
  %tmp27 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat.i8.i, <4 x float> %tmp10, <4 x float> %mul3.i10.i)
  %add.i11.i = fadd <4 x float> %tmp27, %tmp25
  %tmp28 = extractelement <4 x float> %tmp5, i32 2
  %tmp29 = extractelement <4 x float> %mul, i32 2
  %tmp30 = extractelement <4 x float> %tmp5, i32 1
  %tmp31 = extractelement <4 x float> %mul, i32 1
  %tmp32 = extractelement <4 x float> %tmp5, i32 0
  %tmp33 = extractelement <4 x float> %mul, i32 0
  %tmp34 = fmul float %tmp32, %tmp33
  %tmp35 = tail call float @llvm.fmuladd.f32(float %tmp30, float %tmp31, float %tmp34)
  %tmp36 = tail call float @llvm.fmuladd.f32(float %tmp28, float %tmp29, float %tmp35)
  %tmp37 = tail call float @llvm.fmuladd.f32(float 0.000000e+00, float 0.000000e+00, float %tmp36)
  %neg.i12.i = fsub float -0.000000e+00, %tmp37
  %tmp38 = tail call float @llvm.fmuladd.f32(float %tmp26, float 0.000000e+00, float %neg.i12.i)
  %tmp39 = insertelement <4 x float> %add.i11.i, float %tmp38, i64 3
  %tmp40 = extractelement <4 x float> %add.i11.i, i64 1
  %tmp41 = extractelement <4 x float> %add.i11.i, i64 2
  %tmp42 = fmul float %tmp41, %sub1.i.i
  %tmp43 = fsub float -0.000000e+00, %tmp42
  %tmp44 = tail call float @llvm.fmuladd.f32(float %tmp40, float %sub3.i.i, float %tmp43)
  %tmp45 = extractelement <4 x float> %add.i11.i, i64 0
  %tmp46 = fmul float %tmp45, %sub3.i.i
  %tmp47 = fsub float -0.000000e+00, %tmp46
  %tmp48 = tail call float @llvm.fmuladd.f32(float %tmp41, float %sub.i.i, float %tmp47)
  %tmp49 = fmul float %tmp40, %sub.i.i
  %tmp50 = fsub float -0.000000e+00, %tmp49
  %tmp51 = tail call float @llvm.fmuladd.f32(float %tmp45, float %sub1.i.i, float %tmp50)
  %tmp52 = insertelement <4 x float> <float undef, float undef, float undef, float 0.000000e+00>, float %tmp44, i32 0
  %tmp53 = insertelement <4 x float> %tmp52, float %tmp48, i32 1
  %tmp54 = insertelement <4 x float> %tmp53, float %tmp51, i32 2
  %splat.splat.i.i = shufflevector <4 x float> %tmp39, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %tmp55 = extractelement <4 x float> %tmp5, i32 3
  %mul3.i.i = fmul <4 x float> %splat.splat.i8.i, %tmp39
  %tmp56 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat.i.i, <4 x float> %vecinit5.i.i, <4 x float> %mul3.i.i)
  %add.i.i = fadd <4 x float> %tmp56, %tmp54
  %tmp57 = extractelement <4 x float> %add.i11.i, i32 2
  %tmp58 = extractelement <4 x float> %add.i11.i, i32 1
  %tmp59 = extractelement <4 x float> %add.i11.i, i32 0
  %tmp60 = fmul float %tmp59, %sub.i.i
  %tmp61 = tail call float @llvm.fmuladd.f32(float %tmp58, float %sub1.i.i, float %tmp60)
  %tmp62 = tail call float @llvm.fmuladd.f32(float %tmp57, float %sub3.i.i, float %tmp61)
  %tmp63 = tail call float @llvm.fmuladd.f32(float 0.000000e+00, float 0.000000e+00, float %tmp62)
  %neg.i.i = fsub float -0.000000e+00, %tmp63
  %tmp64 = tail call float @llvm.fmuladd.f32(float %tmp38, float %tmp55, float %neg.i.i)
  %tmp65 = insertelement <4 x float> %add.i.i, float %tmp64, i64 3
  %mul2 = fmul <4 x float> %tmp3, %tmp65
  %tmp66 = extractelement <4 x float> %tmp1, i64 3
  %mul3 = fmul float %tmp66, %time
  %tmp67 = insertelement <4 x float> %tmp1, float 0.000000e+00, i32 3
  %tmp68 = shufflevector <4 x float> %tmp67, <4 x float> %tmp1, <4 x i32> <i32 0, i32 5, i32 undef, i32 3>
  %vecinit3.i.i = shufflevector <4 x float> %tmp68, <4 x float> %tmp1, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  %tmp69 = fcmp oeq <4 x float> %vecinit3.i.i, zeroinitializer
  %tmp70 = sext <4 x i1> %tmp69 to <4 x i32>
  %tmp71 = shufflevector <4 x i32> %tmp70, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %tmp72 = shufflevector <4 x i32> %tmp70, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %tmp73 = and <2 x i32> %tmp71, %tmp72
  %tmp74 = extractelement <2 x i32> %tmp73, i64 0
  %tmp75 = extractelement <2 x i32> %tmp73, i64 1
  %tmp76 = and i32 %tmp74, %tmp75
  %tmp77 = icmp sgt i32 %tmp76, -1
  br i1 %tmp77, label %bb, label %qtSet.exit

bb:                                               ; preds = %entry
  %tmp78 = extractelement <4 x float> %tmp1, i32 2
  %tmp79 = extractelement <4 x float> %tmp1, i32 1
  %tmp80 = extractelement <4 x float> %tmp1, i32 0
  %tmp81 = fmul float %tmp80, %tmp80
  %tmp82 = tail call float @llvm.fmuladd.f32(float %tmp79, float %tmp79, float %tmp81)
  %tmp83 = tail call float @llvm.fmuladd.f32(float %tmp78, float %tmp78, float %tmp82)
  %tmp84 = tail call float @llvm.fmuladd.f32(float 0.000000e+00, float 0.000000e+00, float %tmp83)
  %tmp85 = fcmp olt float %tmp84, 0x3810000000000000
  br i1 %tmp85, label %bb86, label %bb96

bb86:                                             ; preds = %bb
  %tmp87 = fmul <4 x float> %vecinit3.i.i, <float 0x4550000000000000, float 0x4550000000000000, float 0x4550000000000000, float 0x4550000000000000>
  %tmp88 = extractelement <4 x float> %tmp87, i64 3
  %tmp89 = extractelement <4 x float> %tmp87, i64 2
  %tmp90 = extractelement <4 x float> %tmp87, i64 1
  %tmp91 = extractelement <4 x float> %tmp87, i64 0
  %tmp92 = fmul float %tmp91, %tmp91
  %tmp93 = tail call float @llvm.fmuladd.f32(float %tmp90, float %tmp90, float %tmp92)
  %tmp94 = tail call float @llvm.fmuladd.f32(float %tmp89, float %tmp89, float %tmp93)
  %tmp95 = tail call float @llvm.fmuladd.f32(float %tmp88, float %tmp88, float %tmp94)
  br label %bb141

bb96:                                             ; preds = %bb
  %tmp97 = fcmp oeq float %tmp84, 0x7FF0000000000000
  br i1 %tmp97, label %bb98, label %bb141

bb98:                                             ; preds = %bb96
  %tmp99 = fmul <4 x float> %vecinit3.i.i, <float 0x3BD0000000000000, float 0x3BD0000000000000, float 0x3BD0000000000000, float 0x3BD0000000000000>
  %tmp100 = extractelement <4 x float> %tmp99, i64 3
  %tmp101 = extractelement <4 x float> %tmp99, i64 2
  %tmp102 = extractelement <4 x float> %tmp99, i64 1
  %tmp103 = extractelement <4 x float> %tmp99, i64 0
  %tmp104 = fmul float %tmp103, %tmp103
  %tmp105 = tail call float @llvm.fmuladd.f32(float %tmp102, float %tmp102, float %tmp104)
  %tmp106 = tail call float @llvm.fmuladd.f32(float %tmp101, float %tmp101, float %tmp105)
  %tmp107 = tail call float @llvm.fmuladd.f32(float %tmp100, float %tmp100, float %tmp106)
  %tmp108 = fcmp oeq float %tmp107, 0x7FF0000000000000
  br i1 %tmp108, label %bb109, label %bb141

bb109:                                            ; preds = %bb98
  %tmp110 = tail call zeroext i1 @llvm.amdgcn.class.f32(float %tmp103, i32 516)
  %tmp111 = sext i1 %tmp110 to i32
  %tmp112 = insertelement <4 x i32> undef, i32 %tmp111, i32 0
  %tmp113 = tail call zeroext i1 @llvm.amdgcn.class.f32(float %tmp102, i32 516)
  %tmp114 = sext i1 %tmp113 to i32
  %tmp115 = insertelement <4 x i32> %tmp112, i32 %tmp114, i32 1
  %tmp116 = tail call zeroext i1 @llvm.amdgcn.class.f32(float %tmp101, i32 516)
  %tmp117 = sext i1 %tmp116 to i32
  %tmp118 = insertelement <4 x i32> %tmp115, i32 %tmp117, i32 2
  %tmp119 = tail call zeroext i1 @llvm.amdgcn.class.f32(float %tmp100, i32 516)
  %tmp120 = sext i1 %tmp119 to i32
  %tmp121 = insertelement <4 x i32> %tmp118, i32 %tmp120, i32 3
  %tmp122 = ashr <4 x i32> %tmp121, <i32 31, i32 31, i32 31, i32 31>
  %tmp123 = and <4 x i32> %tmp122, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %tmp124 = bitcast <4 x i32> %tmp123 to <4 x float>
  %tmp125 = extractelement <4 x float> %tmp124, i64 0
  %tmp126 = tail call float @llvm.copysign.f32(float %tmp125, float %tmp103)
  %tmp127 = insertelement <4 x float> undef, float %tmp126, i32 0
  %tmp128 = extractelement <4 x float> %tmp124, i64 1
  %tmp129 = tail call float @llvm.copysign.f32(float %tmp128, float %tmp102)
  %tmp130 = insertelement <4 x float> %tmp127, float %tmp129, i32 1
  %tmp131 = extractelement <4 x float> %tmp124, i64 2
  %tmp132 = tail call float @llvm.copysign.f32(float %tmp131, float %tmp101)
  %tmp133 = insertelement <4 x float> %tmp130, float %tmp132, i32 2
  %tmp134 = extractelement <4 x float> %tmp124, i64 3
  %tmp135 = tail call float @llvm.copysign.f32(float %tmp134, float %tmp100)
  %tmp136 = insertelement <4 x float> %tmp133, float %tmp135, i32 3
  %tmp137 = fmul float %tmp126, %tmp126
  %tmp138 = tail call float @llvm.fmuladd.f32(float %tmp129, float %tmp129, float %tmp137)
  %tmp139 = tail call float @llvm.fmuladd.f32(float %tmp132, float %tmp132, float %tmp138)
  %tmp140 = tail call float @llvm.fmuladd.f32(float %tmp135, float %tmp135, float %tmp139)
  br label %bb141

bb141:                                            ; preds = %bb109, %bb98, %bb96, %bb86
  %tmp142 = phi <4 x float> [ %tmp87, %bb86 ], [ %tmp136, %bb109 ], [ %tmp99, %bb98 ], [ %vecinit3.i.i, %bb96 ]
  %tmp143 = phi float [ %tmp95, %bb86 ], [ %tmp140, %bb109 ], [ %tmp107, %bb98 ], [ %tmp84, %bb96 ]
  %tmp144 = tail call float @llvm.amdgcn.rsq.f32(float %tmp143)
  %tmp145 = insertelement <4 x float> undef, float %tmp144, i32 0
  %tmp146 = shufflevector <4 x float> %tmp145, <4 x float> undef, <4 x i32> zeroinitializer
  %tmp147 = fmul <4 x float> %tmp142, %tmp146
  br label %qtSet.exit

qtSet.exit:                                       ; preds = %bb141, %entry
  %tmp148 = phi <4 x float> [ %tmp147, %bb141 ], [ %vecinit3.i.i, %entry ]
  %div.i = fmul float %mul3, 5.000000e-01
  %cmp.i.i = fcmp olt float %div.i, 0x400921CAC0000000
  %cond.i.i = select i1 %cmp.i.i, float 0x401921CAC0000000, float 0.000000e+00
  %add.i18.i = fadd float %div.i, %cond.i.i
  %cmp1.i.i = fcmp ogt float %add.i18.i, 0x400921CAC0000000
  %cond2.i.i = select i1 %cmp1.i.i, float 0x401921CAC0000000, float 0.000000e+00
  %sub.i.i48 = fsub float %add.i18.i, %cond2.i.i
  %mul.i.i = fmul float %sub.i.i48, 0x3FF45F3060000000
  %cmp3.i.i = fcmp olt float %sub.i.i48, 0.000000e+00
  %mul5.i.i = select i1 %cmp3.i.i, float 0x3FD9F02F60000000, float 0xBFD9F02F60000000
  %mul6.i.i = fmul float %sub.i.i48, %mul5.i.i
  %tmp149 = tail call float @llvm.fmuladd.f32(float %mul6.i.i, float %sub.i.i48, float %mul.i.i)
  %cmp8.i.i = fcmp olt float %tmp149, 0.000000e+00
  %cond9.i.i = select i1 %cmp8.i.i, float -1.000000e+00, float 1.000000e+00
  %mul10.i.i = fmul float %tmp149, %cond9.i.i
  %neg.i.i49 = fsub float -0.000000e+00, %tmp149
  %tmp150 = tail call float @llvm.fmuladd.f32(float %mul10.i.i, float %tmp149, float %neg.i.i49)
  %tmp151 = tail call float @llvm.fmuladd.f32(float %tmp150, float 0x3FCCCCCCC0000000, float %tmp149)
  %tmp152 = extractelement <4 x float> %tmp148, i64 0
  %mul.i = fmul float %tmp151, %tmp152
  %tmp153 = insertelement <4 x float> undef, float %mul.i, i64 0
  %tmp154 = extractelement <4 x float> %tmp148, i64 1
  %mul2.i = fmul float %tmp151, %tmp154
  %tmp155 = insertelement <4 x float> %tmp153, float %mul2.i, i64 1
  %tmp156 = extractelement <4 x float> %tmp148, i64 2
  %mul3.i = fmul float %tmp151, %tmp156
  %tmp157 = insertelement <4 x float> %tmp155, float %mul3.i, i64 2
  %add.i.i50 = fadd float %div.i, 0x3FF921CAC0000000
  %cmp.i.i.i = fcmp olt float %add.i.i50, 0x400921CAC0000000
  %cond.i.i.i = select i1 %cmp.i.i.i, float 0x401921CAC0000000, float 0.000000e+00
  %add.i.i.i = fadd float %add.i.i50, %cond.i.i.i
  %cmp1.i.i.i = fcmp ogt float %add.i.i.i, 0x400921CAC0000000
  %cond2.i.i.i = select i1 %cmp1.i.i.i, float 0x401921CAC0000000, float 0.000000e+00
  %sub.i.i.i = fsub float %add.i.i.i, %cond2.i.i.i
  %mul.i.i.i = fmul float %sub.i.i.i, 0x3FF45F3060000000
  %cmp3.i.i.i = fcmp olt float %sub.i.i.i, 0.000000e+00
  %mul5.i.i.i = select i1 %cmp3.i.i.i, float 0x3FD9F02F60000000, float 0xBFD9F02F60000000
  %mul6.i.i.i = fmul float %sub.i.i.i, %mul5.i.i.i
  %tmp158 = tail call float @llvm.fmuladd.f32(float %mul6.i.i.i, float %sub.i.i.i, float %mul.i.i.i)
  %cmp8.i.i.i = fcmp olt float %tmp158, 0.000000e+00
  %cond9.i.i.i = select i1 %cmp8.i.i.i, float -1.000000e+00, float 1.000000e+00
  %mul10.i.i.i = fmul float %tmp158, %cond9.i.i.i
  %neg.i.i.i = fsub float -0.000000e+00, %tmp158
  %tmp159 = tail call float @llvm.fmuladd.f32(float %mul10.i.i.i, float %tmp158, float %neg.i.i.i)
  %tmp160 = tail call float @llvm.fmuladd.f32(float %tmp159, float 0x3FCCCCCCC0000000, float %tmp158)
  %tmp161 = insertelement <4 x float> %tmp157, float %tmp160, i64 3
  %sub.i.i32 = fsub float -0.000000e+00, %mul.i
  %vecinit.i.i33 = insertelement <4 x float> undef, float %sub.i.i32, i32 0
  %sub1.i.i34 = fsub float -0.000000e+00, %mul2.i
  %vecinit2.i.i35 = insertelement <4 x float> %vecinit.i.i33, float %sub1.i.i34, i32 1
  %sub3.i.i36 = fsub float -0.000000e+00, %mul3.i
  %vecinit4.i.i37 = insertelement <4 x float> %vecinit2.i.i35, float %sub3.i.i36, i32 2
  %vecinit5.i.i38 = shufflevector <4 x float> %vecinit4.i.i37, <4 x float> %tmp161, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %tmp162 = insertelement <4 x float> %mul2, float 0.000000e+00, i64 3
  ret <4 x float> %tmp162
}

declare float @llvm.copysign.f32(float, float)

declare i1 @llvm.amdgcn.class.f32(float, i32)

declare float @llvm.amdgcn.rsq.f32(float)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!104, !105, !106}
!opencl.ocl.version = !{!107}
!llvm.ident = !{!108, !109}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !74)
!1 = !DIFile(filename: "tmp.cl", directory: "/home/yaxunl/h/git/llvm/assert")
!2 = !{!3, !27, !37, !42, !46, !51, !55, !68}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "BrdfType", file: !4, line: 1334, size: 32, elements: !5)
!4 = !DIFile(filename: "GraphMaterialSystemKernels1.cl", directory: "/home/yaxunl/h/git/llvm/assert")
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}
!6 = !DIEnumerator(name: "BRDF_DEFAULT", value: 0)
!7 = !DIEnumerator(name: "BRDF_LAMBERT", value: 1)
!8 = !DIEnumerator(name: "BRDF_MICROFACET", value: 2)
!9 = !DIEnumerator(name: "BRDF_REFLECT", value: 3)
!10 = !DIEnumerator(name: "BRDF_REFRACT", value: 4)
!11 = !DIEnumerator(name: "BRDF_EMISSIVE", value: 5)
!12 = !DIEnumerator(name: "BRDF_LAYERED", value: 6)
!13 = !DIEnumerator(name: "BRDF_FUR", value: 7)
!14 = !DIEnumerator(name: "BRDF_DIFFUSE_ORENNAYAR", value: 8)
!15 = !DIEnumerator(name: "BRDF_TRANSPARENT", value: 9)
!16 = !DIEnumerator(name: "BRDF_PHONG", value: 10)
!17 = !DIEnumerator(name: "BRDF_WARD", value: 11)
!18 = !DIEnumerator(name: "BRDF_ASHIKHMIN", value: 12)
!19 = !DIEnumerator(name: "BRDF_MICROFACET_GGX", value: 13)
!20 = !DIEnumerator(name: "BRDF_MICROFACET_REFRACTION", value: 14)
!21 = !DIEnumerator(name: "BRDF_PASSTHROUGH", value: 15)
!22 = !DIEnumerator(name: "BRDF_VOLUME", value: 16)
!23 = !DIEnumerator(name: "BRDF_LAMBERT_REFRACTION", value: 17)
!24 = !DIEnumerator(name: "BRDF_MICROFACET_ANISOTROPIC_REFLECTION", value: 18)
!25 = !DIEnumerator(name: "BRDF_MICROFACET_ANISOTROPIC_REFRACTION", value: 19)
!26 = !DIEnumerator(name: "BRDF_MICROFACET_BECKMANN", value: 20)
!27 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "LightType", file: !4, line: 1470, size: 32, elements: !28)
!28 = !{!29, !30, !31, !32, !33, !34, !35, !36}
!29 = !DIEnumerator(name: "LIGHT_POINT", value: 0)
!30 = !DIEnumerator(name: "LIGHT_SPOT", value: 1)
!31 = !DIEnumerator(name: "LIGHT_DIRECTIONAL", value: 2)
!32 = !DIEnumerator(name: "LIGHT_UNIFORM", value: 3)
!33 = !DIEnumerator(name: "LIGHT_MESH", value: 4)
!34 = !DIEnumerator(name: "LIGHT_IBL", value: 5)
!35 = !DIEnumerator(name: "LIGHT_GONIOPHOTO", value: 6)
!36 = !DIEnumerator(name: "LIGHT_SKY", value: 7)
!37 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !4, line: 1569, size: 32, elements: !38)
!38 = !{!39, !40, !41}
!39 = !DIEnumerator(name: "UG3D_FILTER_NEAREST", value: 0)
!40 = !DIEnumerator(name: "UG3D_FILTER_LINEAR", value: 1)
!41 = !DIEnumerator(name: "UG3D_FILTER_MONOTONIC_CUBIC", value: 2)
!42 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !4, line: 1562, size: 32, elements: !43)
!43 = !{!44, !45}
!44 = !DIEnumerator(name: "VH_CHECKER", value: 0)
!45 = !DIEnumerator(name: "VH_OPENVDB", value: 1)
!46 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !4, line: 1554, size: 32, elements: !47)
!47 = !{!48, !49, !50}
!48 = !DIEnumerator(name: "VOLUME_HOMOGENEOUS", value: 0)
!49 = !DIEnumerator(name: "VOLUME_HETEROGENEOUS", value: 1)
!50 = !DIEnumerator(name: "VOLUME_NONE", value: 65535)
!51 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "VolumeFlags", file: !4, line: 1504, size: 32, elements: !52)
!52 = !{!53, !54}
!53 = !DIEnumerator(name: "VFLAGS_NONE", value: 0)
!54 = !DIEnumerator(name: "VFLAGS_SINGLE_SCATTER", value: 1)
!55 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "OffsetIdx", file: !4, line: 1316, size: 32, elements: !56)
!56 = !{!57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67}
!57 = !DIEnumerator(name: "SCENE_OFFSET_IDX_FACE", value: 0)
!58 = !DIEnumerator(name: "SCENE_OFFSET_IDX_VTX", value: 1)
!59 = !DIEnumerator(name: "SCENE_OFFSET_IDX_N", value: 2)
!60 = !DIEnumerator(name: "SCENE_OFFSET_IDX_TEXCRD", value: 3)
!61 = !DIEnumerator(name: "SCENE_OFFSET_IDX_TEXCRD1", value: 4)
!62 = !DIEnumerator(name: "SCENE_OFFSET_IDX_SUBSHAPE_DATA", value: 5)
!63 = !DIEnumerator(name: "SCENE_OFFSET_IDX_MATERIAL", value: 6)
!64 = !DIEnumerator(name: "SCENE_OFFSET_IDX_MATERIAL_DESC", value: 7)
!65 = !DIEnumerator(name: "SCENE_OFFSET_IDX_GRID3D", value: 8)
!66 = !DIEnumerator(name: "SCENE_OFFSET_IDX_WORLD_VOLUME", value: 9)
!67 = !DIEnumerator(name: "SCENE_OFFSET_INTERVAL", value: 4)
!68 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !4, line: 1972, size: 32, elements: !69)
!69 = !{!70, !71, !72, !73}
!70 = !DIEnumerator(name: "FACE_TRIANGLE", value: 0)
!71 = !DIEnumerator(name: "FACE_LINE_SEGMENT", value: 1)
!72 = !DIEnumerator(name: "FACE_QUAD", value: 2)
!73 = !DIEnumerator(name: "FACE_OTHERS", value: 3)
!74 = !{!75, !77, !80, !83, !84, !87, !88, !102}
!75 = !DIDerivedType(tag: DW_TAG_typedef, name: "u32", file: !4, line: 85, baseType: !76)
!76 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!77 = !DIDerivedType(tag: DW_TAG_typedef, name: "float4", file: !78, line: 127, baseType: !79)
!78 = !DIFile(filename: "/home/yaxunl/h/driver/1490442/opencl/include/opencl-c.h", directory: "/home/yaxunl/h/git/llvm/assert")
!79 = !DICompositeType(tag: DW_TAG_array_type, baseType: !80, size: 128, flags: DIFlagVector, elements: !81)
!80 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!81 = !{!82}
!82 = !DISubrange(count: 4)
!83 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!84 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !85, size: 32, addressSpace: 1)
!85 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !86)
!86 = !DIBasicType(name: "half", size: 16, encoding: DW_ATE_float)
!87 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !86, size: 32, addressSpace: 1)
!88 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !89, size: 64)
!89 = !DIDerivedType(tag: DW_TAG_typedef, name: "Face", file: !4, line: 1993, baseType: !90)
!90 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !4, line: 1981, size: 640, elements: !91)
!91 = !{!92, !94, !95, !96, !97, !98, !100, !101}
!92 = !DIDerivedType(tag: DW_TAG_member, name: "m_v", scope: !90, file: !4, line: 1983, baseType: !93, size: 128)
!93 = !DICompositeType(tag: DW_TAG_array_type, baseType: !83, size: 128, elements: !81)
!94 = !DIDerivedType(tag: DW_TAG_member, name: "m_n", scope: !90, file: !4, line: 1984, baseType: !93, size: 128, offset: 128)
!95 = !DIDerivedType(tag: DW_TAG_member, name: "m_t", scope: !90, file: !4, line: 1985, baseType: !93, size: 128, offset: 256)
!96 = !DIDerivedType(tag: DW_TAG_member, name: "m_t1", scope: !90, file: !4, line: 1986, baseType: !93, size: 128, offset: 384)
!97 = !DIDerivedType(tag: DW_TAG_member, name: "m_m", scope: !90, file: !4, line: 1988, baseType: !83, size: 32, offset: 512)
!98 = !DIDerivedType(tag: DW_TAG_member, name: "m_type", scope: !90, file: !4, line: 1989, baseType: !99, size: 32, offset: 544)
!99 = !DIDerivedType(tag: DW_TAG_typedef, name: "FaceType", file: !4, line: 1978, baseType: !68)
!100 = !DIDerivedType(tag: DW_TAG_member, name: "m_lightIdx", scope: !90, file: !4, line: 1990, baseType: !83, size: 32, offset: 576)
!101 = !DIDerivedType(tag: DW_TAG_member, name: "m_padding", scope: !90, file: !4, line: 1991, baseType: !83, size: 32, offset: 608)
!102 = !DIDerivedType(tag: DW_TAG_typedef, name: "u64", file: !4, line: 84, baseType: !103)
!103 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!104 = !{i32 2, !"Dwarf Version", i32 2}
!105 = !{i32 2, !"Debug Info Version", i32 3}
!106 = !{i32 1, !"wchar_size", i32 4}
!107 = !{i32 2, i32 0}
!108 = !{!"clang version 7.0.0"}
!109 = !{!"clang version 4.0 "}
!110 = distinct !DISubprogram(name: "Scene_transformT", scope: !4, file: !4, line: 2182, type: !111, isLocal: false, isDefinition: true, scopeLine: 2183, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !116)
!111 = !DISubroutineType(types: !112)
!112 = !{!77, !83, !77, !80, !113, !115}
!113 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !114, size: 64)
!114 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!115 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !75, size: 64)
!116 = !{!117, !118, !119, !120, !121, !122, !147, !149, !150, !151, !152, !153}
!117 = !DILocalVariable(name: "subshapeIdx", arg: 1, scope: !110, file: !4, line: 2182, type: !83)
!118 = !DILocalVariable(name: "v", arg: 2, scope: !110, file: !4, line: 2182, type: !77)
!119 = !DILocalVariable(name: "time", arg: 3, scope: !110, file: !4, line: 2182, type: !80)
!120 = !DILocalVariable(name: "gScene", arg: 4, scope: !110, file: !4, line: 2182, type: !113)
!121 = !DILocalVariable(name: "gSceneOffsets", arg: 5, scope: !110, file: !4, line: 2182, type: !115)
!122 = !DILocalVariable(name: "ss", scope: !110, file: !4, line: 2184, type: !123)
!123 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !124, size: 64)
!124 = !DIDerivedType(tag: DW_TAG_typedef, name: "ShapeData", file: !4, line: 1949, baseType: !125)
!125 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !4, line: 1927, size: 1024, elements: !126)
!126 = !{!127, !128, !130, !131, !132, !133, !134, !135, !136, !137, !138, !141, !142, !145, !146}
!127 = !DIDerivedType(tag: DW_TAG_member, name: "m_translation", scope: !125, file: !4, line: 1929, baseType: !77, size: 128)
!128 = !DIDerivedType(tag: DW_TAG_member, name: "m_quaternion", scope: !125, file: !4, line: 1930, baseType: !129, size: 128, offset: 128)
!129 = !DIDerivedType(tag: DW_TAG_typedef, name: "Quaternion", file: !4, line: 625, baseType: !77)
!130 = !DIDerivedType(tag: DW_TAG_member, name: "m_linearMotion", scope: !125, file: !4, line: 1931, baseType: !77, size: 128, offset: 256)
!131 = !DIDerivedType(tag: DW_TAG_member, name: "m_angularMotion", scope: !125, file: !4, line: 1932, baseType: !77, size: 128, offset: 384)
!132 = !DIDerivedType(tag: DW_TAG_member, name: "m_scaleMotion", scope: !125, file: !4, line: 1933, baseType: !77, size: 128, offset: 512)
!133 = !DIDerivedType(tag: DW_TAG_member, name: "m_rootIdx", scope: !125, file: !4, line: 1935, baseType: !75, size: 32, offset: 640)
!134 = !DIDerivedType(tag: DW_TAG_member, name: "m_materialIdx", scope: !125, file: !4, line: 1936, baseType: !75, size: 32, offset: 672)
!135 = !DIDerivedType(tag: DW_TAG_member, name: "m_shapeAddr", scope: !125, file: !4, line: 1937, baseType: !102, size: 64, offset: 704)
!136 = !DIDerivedType(tag: DW_TAG_member, name: "m_scale", scope: !125, file: !4, line: 1939, baseType: !77, size: 128, offset: 768)
!137 = !DIDerivedType(tag: DW_TAG_member, name: "m_flags", scope: !125, file: !4, line: 1941, baseType: !75, size: 32, offset: 896)
!138 = !DIDerivedType(tag: DW_TAG_member, name: "m_hasUv", scope: !125, file: !4, line: 1942, baseType: !139, size: 8, offset: 928)
!139 = !DIDerivedType(tag: DW_TAG_typedef, name: "u8", file: !4, line: 87, baseType: !140)
!140 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!141 = !DIDerivedType(tag: DW_TAG_member, name: "m_padd00", scope: !125, file: !4, line: 1943, baseType: !139, size: 8, offset: 936)
!142 = !DIDerivedType(tag: DW_TAG_member, name: "m_objectGroupId", scope: !125, file: !4, line: 1944, baseType: !143, size: 16, offset: 944)
!143 = !DIDerivedType(tag: DW_TAG_typedef, name: "u16", file: !4, line: 86, baseType: !144)
!144 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!145 = !DIDerivedType(tag: DW_TAG_member, name: "m_faceIndexShift", scope: !125, file: !4, line: 1946, baseType: !75, size: 32, offset: 960)
!146 = !DIDerivedType(tag: DW_TAG_member, name: "m_padd2", scope: !125, file: !4, line: 1947, baseType: !75, size: 32, offset: 992)
!147 = !DILocalVariable(name: "mt", scope: !148, file: !4, line: 2187, type: !77)
!148 = distinct !DILexicalBlock(scope: !110, file: !4, line: 2186, column: 2)
!149 = !DILocalVariable(name: "ma", scope: !148, file: !4, line: 2188, type: !77)
!150 = !DILocalVariable(name: "ms", scope: !148, file: !4, line: 2189, type: !77)
!151 = !DILocalVariable(name: "t", scope: !148, file: !4, line: 2191, type: !77)
!152 = !DILocalVariable(name: "q", scope: !148, file: !4, line: 2193, type: !129)
!153 = !DILocalVariable(name: "w", scope: !148, file: !4, line: 2194, type: !80)
!154 = !DILocation(line: 2182, column: 80, scope: !110)
!155 = !DILocation(line: 2182, column: 102, scope: !110)
