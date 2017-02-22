; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}phi1:
; CHECK: s_buffer_load_dword [[DST:s[0-9]]], {{s\[[0-9]+:[0-9]+\]}}, 0x0
; CHECK: v_mov_b32_e32 v{{[0-9]}}, [[DST]]
define amdgpu_ps void @phi1(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <8 x i32> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 0)
  %tmp22 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 16)
  %tmp23 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 32)
  %tmp24 = fptosi float %tmp22 to i32
  %tmp25 = icmp ne i32 %tmp24, 0
  br i1 %tmp25, label %ENDIF, label %ELSE

ELSE:                                             ; preds = %main_body
  %tmp26 = fsub float -0.000000e+00, %tmp21
  br label %ENDIF

ENDIF:                                            ; preds = %ELSE, %main_body
  %temp.0 = phi float [ %tmp26, %ELSE ], [ %tmp21, %main_body ]
  %tmp27 = fadd float %temp.0, %tmp23
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp27, float %tmp27, float 0.000000e+00, float 1.000000e+00, i1 true, i1 true) #0
  ret void
}

; Make sure this program doesn't crash
; CHECK-LABEL: {{^}}phi2:
define amdgpu_ps void @phi2(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <8 x i32> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #1 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 16)
  %tmp22 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 32)
  %tmp23 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 36)
  %tmp24 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 40)
  %tmp25 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 48)
  %tmp26 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 52)
  %tmp27 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 56)
  %tmp28 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 64)
  %tmp29 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 68)
  %tmp30 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 72)
  %tmp31 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 76)
  %tmp32 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 80)
  %tmp33 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 84)
  %tmp34 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 88)
  %tmp35 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 92)
  %tmp36 = getelementptr <8 x i32>, <8 x i32> addrspace(2)* %arg2, i32 0
  %tmp37 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp36, !tbaa !0
  %tmp38 = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg1, i32 0
  %tmp39 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp38, !tbaa !0
  %i.i = extractelement <2 x i32> %arg5, i32 0
  %j.i = extractelement <2 x i32> %arg5, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 0, i32 0, i32 %arg3) #1
  %p2.i = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 0, i32 0, i32 %arg3) #1
  %i.i19 = extractelement <2 x i32> %arg5, i32 0
  %j.i20 = extractelement <2 x i32> %arg5, i32 1
  %i.f.i21 = bitcast i32 %i.i19 to float
  %j.f.i22 = bitcast i32 %j.i20 to float
  %p1.i23 = call float @llvm.amdgcn.interp.p1(float %i.f.i21, i32 1, i32 0, i32 %arg3) #1
  %p2.i24 = call float @llvm.amdgcn.interp.p2(float %p1.i23, float %j.f.i22, i32 1, i32 0, i32 %arg3) #1
  %i.i13 = extractelement <2 x i32> %arg5, i32 0
  %j.i14 = extractelement <2 x i32> %arg5, i32 1
  %i.f.i15 = bitcast i32 %i.i13 to float
  %j.f.i16 = bitcast i32 %j.i14 to float
  %p1.i17 = call float @llvm.amdgcn.interp.p1(float %i.f.i15, i32 0, i32 1, i32 %arg3) #1
  %p2.i18 = call float @llvm.amdgcn.interp.p2(float %p1.i17, float %j.f.i16, i32 0, i32 1, i32 %arg3) #1
  %i.i7 = extractelement <2 x i32> %arg5, i32 0
  %j.i8 = extractelement <2 x i32> %arg5, i32 1
  %i.f.i9 = bitcast i32 %i.i7 to float
  %j.f.i10 = bitcast i32 %j.i8 to float
  %p1.i11 = call float @llvm.amdgcn.interp.p1(float %i.f.i9, i32 1, i32 1, i32 %arg3) #1
  %p2.i12 = call float @llvm.amdgcn.interp.p2(float %p1.i11, float %j.f.i10, i32 1, i32 1, i32 %arg3) #1
  %i.i1 = extractelement <2 x i32> %arg5, i32 0
  %j.i2 = extractelement <2 x i32> %arg5, i32 1
  %i.f.i3 = bitcast i32 %i.i1 to float
  %j.f.i4 = bitcast i32 %j.i2 to float
  %p1.i5 = call float @llvm.amdgcn.interp.p1(float %i.f.i3, i32 2, i32 1, i32 %arg3) #1
  %p2.i6 = call float @llvm.amdgcn.interp.p2(float %p1.i5, float %j.f.i4, i32 2, i32 1, i32 %arg3) #1
  %tmp45 = bitcast float %p2.i to i32
  %tmp46 = bitcast float %p2.i24 to i32
  %tmp47 = insertelement <2 x i32> undef, i32 %tmp45, i32 0
  %tmp48 = insertelement <2 x i32> %tmp47, i32 %tmp46, i32 1
  %tmp39.bc = bitcast <16 x i8> %tmp39 to <4 x i32>
  %tmp49 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> %tmp48, <8 x i32> %tmp37, <4 x i32> %tmp39.bc, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp50 = extractelement <4 x float> %tmp49, i32 2
  %tmp51 = call float @llvm.fabs.f32(float %tmp50)
  %tmp52 = fmul float %p2.i18, %p2.i18
  %tmp53 = fmul float %p2.i12, %p2.i12
  %tmp54 = fadd float %tmp53, %tmp52
  %tmp55 = fmul float %p2.i6, %p2.i6
  %tmp56 = fadd float %tmp54, %tmp55
  %tmp57 = call float @llvm.amdgcn.rsq.f32(float %tmp56)
  %tmp58 = fmul float %p2.i18, %tmp57
  %tmp59 = fmul float %p2.i12, %tmp57
  %tmp60 = fmul float %p2.i6, %tmp57
  %tmp61 = fmul float %tmp58, %tmp22
  %tmp62 = fmul float %tmp59, %tmp23
  %tmp63 = fadd float %tmp62, %tmp61
  %tmp64 = fmul float %tmp60, %tmp24
  %tmp65 = fadd float %tmp63, %tmp64
  %tmp66 = fsub float -0.000000e+00, %tmp25
  %tmp67 = fmul float %tmp65, %tmp51
  %tmp68 = fadd float %tmp67, %tmp66
  %tmp69 = fmul float %tmp26, %tmp68
  %tmp70 = fmul float %tmp27, %tmp68
  %tmp71 = call float @llvm.fabs.f32(float %tmp69)
  %tmp72 = fcmp olt float 0x3EE4F8B580000000, %tmp71
  %tmp73 = sext i1 %tmp72 to i32
  %tmp74 = bitcast i32 %tmp73 to float
  %tmp75 = bitcast float %tmp74 to i32
  %tmp76 = icmp ne i32 %tmp75, 0
  br i1 %tmp76, label %IF, label %ENDIF

IF:                                               ; preds = %main_body
  %tmp77 = fsub float -0.000000e+00, %tmp69
  %tmp78 = call float @llvm.exp2.f32(float %tmp77)
  %tmp79 = fsub float -0.000000e+00, %tmp78
  %tmp80 = fadd float 1.000000e+00, %tmp79
  %tmp81 = fdiv float 1.000000e+00, %tmp69
  %tmp82 = fmul float %tmp80, %tmp81
  %tmp83 = fmul float %tmp31, %tmp82
  br label %ENDIF

ENDIF:                                            ; preds = %IF, %main_body
  %temp4.0 = phi float [ %tmp83, %IF ], [ %tmp31, %main_body ]
  %tmp84 = call float @llvm.fabs.f32(float %tmp70)
  %tmp85 = fcmp olt float 0x3EE4F8B580000000, %tmp84
  %tmp86 = sext i1 %tmp85 to i32
  %tmp87 = bitcast i32 %tmp86 to float
  %tmp88 = bitcast float %tmp87 to i32
  %tmp89 = icmp ne i32 %tmp88, 0
  br i1 %tmp89, label %IF25, label %ENDIF24

IF25:                                             ; preds = %ENDIF
  %tmp90 = fsub float -0.000000e+00, %tmp70
  %tmp91 = call float @llvm.exp2.f32(float %tmp90)
  %tmp92 = fsub float -0.000000e+00, %tmp91
  %tmp93 = fadd float 1.000000e+00, %tmp92
  %tmp94 = fdiv float 1.000000e+00, %tmp70
  %tmp95 = fmul float %tmp93, %tmp94
  %tmp96 = fmul float %tmp35, %tmp95
  br label %ENDIF24

ENDIF24:                                          ; preds = %IF25, %ENDIF
  %temp8.0 = phi float [ %tmp96, %IF25 ], [ %tmp35, %ENDIF ]
  %tmp97 = fmul float %tmp28, %temp4.0
  %tmp98 = fmul float %tmp29, %temp4.0
  %tmp99 = fmul float %tmp30, %temp4.0
  %tmp100 = fmul float %tmp32, %temp8.0
  %tmp101 = fadd float %tmp100, %tmp97
  %tmp102 = fmul float %tmp33, %temp8.0
  %tmp103 = fadd float %tmp102, %tmp98
  %tmp104 = fmul float %tmp34, %temp8.0
  %tmp105 = fadd float %tmp104, %tmp99
  %tmp106 = call float @llvm.pow.f32(float %tmp51, float %tmp21)
  %tmp107 = fsub float -0.000000e+00, %tmp101
  %tmp108 = fmul float %tmp107, %tmp106
  %tmp109 = fsub float -0.000000e+00, %tmp103
  %tmp110 = fmul float %tmp109, %tmp106
  %tmp111 = fsub float -0.000000e+00, %tmp105
  %tmp112 = fmul float %tmp111, %tmp106
  %tmp113 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp108, float %tmp110)
  %tmp115 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp112, float 1.000000e+00)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %tmp113, <2 x half> %tmp115, i1 true, i1 true) #0
  ret void
}

; We just want ot make sure the program doesn't crash
; CHECK-LABEL: {{^}}loop:
define amdgpu_ps void @loop(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <8 x i32> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 0)
  %tmp22 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 4)
  %tmp23 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 8)
  %tmp24 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 12)
  %tmp25 = fptosi float %tmp24 to i32
  %tmp26 = bitcast i32 %tmp25 to float
  %tmp27 = bitcast float %tmp26 to i32
  br label %LOOP

LOOP:                                             ; preds = %ENDIF, %main_body
  %temp4.0 = phi float [ %tmp21, %main_body ], [ %temp5.0, %ENDIF ]
  %temp5.0 = phi float [ %tmp22, %main_body ], [ %temp6.0, %ENDIF ]
  %temp6.0 = phi float [ %tmp23, %main_body ], [ %temp4.0, %ENDIF ]
  %temp8.0 = phi float [ 0.000000e+00, %main_body ], [ %tmp36, %ENDIF ]
  %tmp28 = bitcast float %temp8.0 to i32
  %tmp29 = icmp sge i32 %tmp28, %tmp27
  %tmp30 = sext i1 %tmp29 to i32
  %tmp31 = bitcast i32 %tmp30 to float
  %tmp32 = bitcast float %tmp31 to i32
  %tmp33 = icmp ne i32 %tmp32, 0
  br i1 %tmp33, label %IF, label %ENDIF

IF:                                               ; preds = %LOOP
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %temp4.0, float %temp5.0, float %temp6.0, float 1.000000e+00, i1 true, i1 true) #0
  ret void

ENDIF:                                            ; preds = %LOOP
  %tmp34 = bitcast float %temp8.0 to i32
  %tmp35 = add i32 %tmp34, 1
  %tmp36 = bitcast i32 %tmp35 to float
  br label %LOOP
}

; This checks for a bug in the FixSGPRCopies pass where VReg96
; registers were being identified as an SGPR regclass which was causing
; an assertion failure.

; CHECK-LABEL: {{^}}sample_v3:
; CHECK: v_mov_b32_e32 v[[SAMPLE_LO:[0-9]+]], 11
; CHECK: v_mov_b32_e32 v[[SAMPLE_HI:[0-9]+]], 13
; CHECK: s_branch

; CHECK-DAG: v_mov_b32_e32 v[[SAMPLE_LO:[0-9]+]], 5
; CHECK-DAG: v_mov_b32_e32 v[[SAMPLE_HI:[0-9]+]], 7

; CHECK: BB{{[0-9]+_[0-9]+}}:
; CHECK: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[SAMPLE_LO]]:[[SAMPLE_HI]]{{\]}}
; CHECK: exp
; CHECK: s_endpgm
define amdgpu_ps void @sample_v3([17 x <16 x i8>] addrspace(2)* byval %arg, [32 x <16 x i8>] addrspace(2)* byval %arg1, [16 x <8 x i32>] addrspace(2)* byval %arg2, float inreg %arg3, i32 inreg %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <3 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, float %arg20) #0 {
entry:
  %tmp = getelementptr [17 x <16 x i8>], [17 x <16 x i8>] addrspace(2)* %arg, i64 0, i32 0
  %tmp21 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp22 = call float @llvm.SI.load.const(<16 x i8> %tmp21, i32 16)
  %tmp23 = getelementptr [16 x <8 x i32>], [16 x <8 x i32>] addrspace(2)* %arg2, i64 0, i32 0
  %tmp24 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp23, !tbaa !0
  %tmp25 = getelementptr [32 x <16 x i8>], [32 x <16 x i8>] addrspace(2)* %arg1, i64 0, i32 0
  %tmp26 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp25, !tbaa !0
  %tmp27 = fcmp oeq float %tmp22, 0.000000e+00
  %tmp26.bc = bitcast <16 x i8> %tmp26 to <4 x i32>
  br i1 %tmp27, label %if, label %else

if:                                               ; preds = %entry
  %val.if = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> <i32 11, i32 13>, <8 x i32> %tmp24, <4 x i32> %tmp26.bc, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %val.if.0 = extractelement <4 x float> %val.if, i32 0
  %val.if.1 = extractelement <4 x float> %val.if, i32 1
  %val.if.2 = extractelement <4 x float> %val.if, i32 2
  br label %endif

else:                                             ; preds = %entry
  %val.else = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> <i32 5, i32 7>, <8 x i32> %tmp24, <4 x i32> %tmp26.bc, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %val.else.0 = extractelement <4 x float> %val.else, i32 0
  %val.else.1 = extractelement <4 x float> %val.else, i32 1
  %val.else.2 = extractelement <4 x float> %val.else, i32 2
  br label %endif

endif:                                            ; preds = %else, %if
  %val.0 = phi float [ %val.if.0, %if ], [ %val.else.0, %else ]
  %val.1 = phi float [ %val.if.1, %if ], [ %val.else.1, %else ]
  %val.2 = phi float [ %val.if.2, %if ], [ %val.else.2, %else ]
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %val.0, float %val.1, float %val.2, float 0.000000e+00, i1 true, i1 true) #0
  ret void
}

; CHECK-LABEL: {{^}}copy1:
; CHECK: buffer_load_dword
; CHECK: v_add
; CHECK: s_endpgm
define void @copy1(float addrspace(1)* %out, float addrspace(1)* %in0) {
entry:
  %tmp = load float, float addrspace(1)* %in0
  %tmp1 = fcmp oeq float %tmp, 0.000000e+00
  br i1 %tmp1, label %if0, label %endif

if0:                                              ; preds = %entry
  %tmp2 = bitcast float %tmp to i32
  %tmp3 = fcmp olt float %tmp, 0.000000e+00
  br i1 %tmp3, label %if1, label %endif

if1:                                              ; preds = %if0
  %tmp4 = add i32 %tmp2, 1
  br label %endif

endif:                                            ; preds = %if1, %if0, %entry
  %tmp5 = phi i32 [ 0, %entry ], [ %tmp2, %if0 ], [ %tmp4, %if1 ]
  %tmp6 = bitcast i32 %tmp5 to float
  store float %tmp6, float addrspace(1)* %out
  ret void
}

; This test is just checking that we don't crash / assertion fail.
; CHECK-LABEL: {{^}}copy2:
; CHECK: s_endpgm
define amdgpu_ps void @copy2([17 x <16 x i8>] addrspace(2)* byval %arg, [32 x <16 x i8>] addrspace(2)* byval %arg1, [16 x <8 x i32>] addrspace(2)* byval %arg2, float inreg %arg3, i32 inreg %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <3 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, float %arg20) #0 {
entry:
  br label %LOOP68

LOOP68:                                           ; preds = %ENDIF69, %entry
  %temp4.7 = phi float [ 0.000000e+00, %entry ], [ %v, %ENDIF69 ]
  %t = phi i32 [ 20, %entry ], [ %x, %ENDIF69 ]
  %g = icmp eq i32 0, %t
  %l = bitcast float %temp4.7 to i32
  br i1 %g, label %IF70, label %ENDIF69

IF70:                                             ; preds = %LOOP68
  %q = icmp ne i32 %l, 13
  %temp.8 = select i1 %q, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %temp.8, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, i1 true, i1 true) #0
  ret void

ENDIF69:                                          ; preds = %LOOP68
  %u = add i32 %l, %t
  %v = bitcast i32 %u to float
  %x = add i32 %t, -1
  br label %LOOP68
}

; This test checks that image_sample resource descriptors aren't loaded into
; vgprs.  The verifier will fail if this happens.
; CHECK-LABEL:{{^}}sample_rsrc

; CHECK: s_cmp_eq_u32
; CHECK: s_cbranch_scc0 [[END:BB[0-9]+_[0-9]+]]

; CHECK: v_add_i32_e32 v[[ADD:[0-9]+]], vcc, 1, v{{[0-9]+}}

; [[END]]:
; CHECK: image_sample v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+}}:[[ADD]]{{\]}}
; CHECK: s_endpgm
define amdgpu_ps void @sample_rsrc([6 x <16 x i8>] addrspace(2)* byval %arg, [17 x <16 x i8>] addrspace(2)* byval %arg1, [16 x <4 x i32>] addrspace(2)* byval %arg2, [32 x <8 x i32>] addrspace(2)* byval %arg3, float inreg %arg4, i32 inreg %arg5, <2 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <3 x i32> %arg9, <2 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, i32 %arg19, float %arg20, float %arg21) #0 {
bb:
  %tmp = getelementptr [17 x <16 x i8>], [17 x <16 x i8>] addrspace(2)* %arg1, i32 0, i32 0
  %tmp22 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !3
  %tmp23 = call float @llvm.SI.load.const(<16 x i8> %tmp22, i32 16)
  %tmp25 = getelementptr [32 x <8 x i32>], [32 x <8 x i32>] addrspace(2)* %arg3, i32 0, i32 0
  %tmp26 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp25, !tbaa !3
  %tmp27 = getelementptr [16 x <4 x i32>], [16 x <4 x i32>] addrspace(2)* %arg2, i32 0, i32 0
  %tmp28 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp27, !tbaa !3
  %i.i = extractelement <2 x i32> %arg7, i32 0
  %j.i = extractelement <2 x i32> %arg7, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 0, i32 0, i32 %arg5) #0
  %p2.i = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 0, i32 0, i32 %arg5) #0
  %i.i1 = extractelement <2 x i32> %arg7, i32 0
  %j.i2 = extractelement <2 x i32> %arg7, i32 1
  %i.f.i3 = bitcast i32 %i.i1 to float
  %j.f.i4 = bitcast i32 %j.i2 to float
  %p1.i5 = call float @llvm.amdgcn.interp.p1(float %i.f.i3, i32 1, i32 0, i32 %arg5) #0
  %p2.i6 = call float @llvm.amdgcn.interp.p2(float %p1.i5, float %j.f.i4, i32 1, i32 0, i32 %arg5) #0
  %tmp31 = bitcast float %tmp23 to i32
  %tmp36 = icmp ne i32 %tmp31, 0
  br i1 %tmp36, label %bb38, label %bb80

bb38:                                             ; preds = %bb
  %tmp52 = bitcast float %p2.i to i32
  %tmp53 = bitcast float %p2.i6 to i32
  %tmp54 = insertelement <2 x i32> undef, i32 %tmp52, i32 0
  %tmp55 = insertelement <2 x i32> %tmp54, i32 %tmp53, i32 1
  %tmp56 = bitcast <8 x i32> %tmp26 to <8 x i32>
  %tmp58 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> %tmp55, <8 x i32> %tmp56, <4 x i32> %tmp28, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  br label %bb71

bb80:                                             ; preds = %bb
  %tmp81 = bitcast float %p2.i to i32
  %tmp82 = bitcast float %p2.i6 to i32
  %tmp82.2 = add i32 %tmp82, 1
  %tmp83 = insertelement <2 x i32> undef, i32 %tmp81, i32 0
  %tmp84 = insertelement <2 x i32> %tmp83, i32 %tmp82.2, i32 1
  %tmp85 = bitcast <8 x i32> %tmp26 to <8 x i32>
  %tmp87 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> %tmp84, <8 x i32> %tmp85, <4 x i32> %tmp28, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  br label %bb71

bb71:                                             ; preds = %bb80, %bb38
  %tmp72 = phi <4 x float> [ %tmp58, %bb38 ], [ %tmp87, %bb80 ]
  %tmp88 = extractelement <4 x float> %tmp72, i32 0
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp88, float %tmp88, float %tmp88, float %tmp88, i1 true, i1 true) #0
  ret void
}

; Check the the resource descriptor is stored in an sgpr.
; CHECK-LABEL: {{^}}mimg_srsrc_sgpr:
; CHECK: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x1
define amdgpu_ps void @mimg_srsrc_sgpr([34 x <8 x i32>] addrspace(2)* byval %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp7 = getelementptr [34 x <8 x i32>], [34 x <8 x i32>] addrspace(2)* %arg, i32 0, i32 %tid
  %tmp8 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp7, align 32, !tbaa !0
  %tmp9 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> <i32 1061158912, i32 1048576000>, <8 x i32> %tmp8, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp10 = extractelement <4 x float> %tmp9, i32 0
  %tmp12 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float undef, float %tmp10)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %tmp12, <2 x half> undef, i1 true, i1 true) #0
  ret void
}

; Check the the sampler is stored in an sgpr.
; CHECK-LABEL: {{^}}mimg_ssamp_sgpr:
; CHECK: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x1
define amdgpu_ps void @mimg_ssamp_sgpr([17 x <4 x i32>] addrspace(2)* byval %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp7 = getelementptr [17 x <4 x i32>], [17 x <4 x i32>] addrspace(2)* %arg, i32 0, i32 %tid
  %tmp8 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp7, align 16, !tbaa !0
  %tmp9 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> <i32 1061158912, i32 1048576000>, <8 x i32> undef, <4 x i32> %tmp8, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp10 = extractelement <4 x float> %tmp9, i32 0
  %tmp12 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp10, float undef)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %tmp12, <2 x half> undef, i1 true, i1 true) #0
  ret void
}

declare float @llvm.fabs.f32(float) #1
declare float @llvm.amdgcn.rsq.f32(float) #1
declare float @llvm.exp2.f32(float) #1
declare float @llvm.pow.f32(float, float) #1
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #1
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #1
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #0
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1

declare <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", !2}
!2 = !{!"tbaa root"}
!3 = !{!1, !1, i64 0}
