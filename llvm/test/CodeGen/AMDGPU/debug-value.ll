; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s

%struct.wombat = type { [4 x i32], [4 x i32], [4 x i32] }

define amdgpu_kernel void @wobble(i8 addrspace(1)* nocapture readonly %arg) #0 !dbg !4 {
bb:
  %tmp = load i32, i32 addrspace(1)* undef, align 4
  %tmp1 = load <4 x float>, <4 x float> addrspace(1)* undef, align 16
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = shufflevector <4 x float> undef, <4 x float> %tmp1, <2 x i32> <i32 3, i32 7>
  %tmp4 = call float @barney() #2
  %tmp5 = getelementptr inbounds i8, i8 addrspace(1)* %arg, i64 0
  %tmp6 = bitcast i8 addrspace(1)* %tmp5 to <2 x float> addrspace(1)*
  %tmp7 = getelementptr inbounds i8, i8 addrspace(1)* %arg, i64 0
  %tmp8 = bitcast i8 addrspace(1)* %tmp7 to %struct.wombat addrspace(1)*
  %tmp9 = getelementptr inbounds %struct.wombat, %struct.wombat addrspace(1)* %tmp8, i64 %tmp2, i32 2, i64 0
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp11 = sext i32 %tmp10 to i64
  %tmp12 = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %tmp6, i64 %tmp11
  %tmp13 = bitcast <2 x float> addrspace(1)* %tmp12 to i64 addrspace(1)*
  %tmp14 = getelementptr inbounds i8, i8 addrspace(1)* %arg, i64 undef
  %tmp15 = bitcast i8 addrspace(1)* %tmp14 to <4 x float> addrspace(1)*
  %tmp16 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %tmp15, i64 undef
  %tmp17 = load <4 x float>, <4 x float> addrspace(1)* %tmp16, align 16
  %tmp18 = fsub <4 x float> %tmp17, %tmp17
  %ext = extractelement <4 x float> %tmp18, i32 1
  %tmp19 = fadd float %ext, 0.000000e+00
  %tmp20 = fcmp oeq float %tmp19, 0.000000e+00
  br i1 %tmp20, label %bb21, label %bb25

bb21:                                             ; preds = %bb
  %tmp22 = fmul <4 x float> %tmp18, %tmp18
  %tmp23 = fadd <4 x float> %tmp22, %tmp22
  %tmp24 = fmul <4 x float> %tmp23, %tmp23
  br label %bb28

bb25:                                             ; preds = %bb
  %tmp26 = insertelement <4 x float> undef, float 0.000000e+00, i32 1
  %tmp27 = insertelement <4 x float> %tmp26, float undef, i32 2
  br label %bb28

bb28:                                             ; preds = %bb25, %bb21
  %tmp29 = phi <4 x float> [ %tmp27, %bb25 ], [ %tmp24, %bb21 ]
  store <4 x float> %tmp29, <4 x float> addrspace(5)* undef, align 16
  %tmp30 = getelementptr inbounds %struct.wombat, %struct.wombat addrspace(1)* %tmp8, i64 %tmp2, i32 2, i64 2
  %tmp31 = load i32, i32 addrspace(1)* %tmp30, align 4
  %tmp32 = sext i32 %tmp31 to i64
  %tmp33 = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %tmp6, i64 %tmp32
  %tmp34 = bitcast <2 x float> addrspace(1)* %tmp33 to i64 addrspace(1)*
  %tmp35 = load i64, i64 addrspace(1)* %tmp34, align 8
  %tmp36 = load i32, i32 addrspace(1)* undef, align 4
  %tmp37 = sext i32 %tmp36 to i64
  %tmp38 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* null, i64 %tmp37
  %tmp39 = load <4 x float>, <4 x float> addrspace(1)* %tmp38, align 16
  %tmp40 = load <4 x float>, <4 x float> addrspace(1)* undef, align 16
  %tmp41 = fsub <4 x float> zeroinitializer, %tmp40
  %tmp42 = fsub <4 x float> %tmp39, %tmp40
  %tmp43 = extractelement <4 x float> %tmp40, i32 1
  %tmp44 = fsub float %tmp43, undef
  %tmp45 = fadd float undef, undef
  %tmp46 = fdiv float %tmp44, %tmp45
  %tmp47 = insertelement <4 x float> undef, float %tmp46, i32 0
  %tmp48 = shufflevector <4 x float> %tmp47, <4 x float> undef, <4 x i32> zeroinitializer
  %tmp49 = fsub <4 x float> %tmp48, %tmp40
  %tmp50 = extractelement <4 x float> %tmp41, i32 1
  %tmp51 = extractelement <4 x float> %tmp42, i32 2
  %tmp52 = fmul float undef, undef
  %tmp53 = fadd float %tmp52, undef
  %tmp54 = fadd float %tmp51, %tmp53
  %tmp55 = extractelement <4 x float> %tmp49, i32 1
  %tmp56 = fmul float %tmp55, %tmp50
  %tmp57 = fmul float %tmp54, %tmp56
  %tmp58 = fdiv float %tmp57, 0.000000e+00
  ; CHECK: ;DEBUG_VALUE: foo:var <- [DW_OP_constu 1, DW_OP_swap, DW_OP_xderef]
  call void @llvm.dbg.value(metadata <4 x float> %tmp29, metadata !3, metadata !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)) #2, !dbg !5
  %tmp59 = bitcast i64 %tmp35 to <2 x float>
  %tmp60 = insertelement <2 x float> undef, float %tmp58, i32 0
  %tmp61 = shufflevector <2 x float> %tmp60, <2 x float> undef, <2 x i32> zeroinitializer
  %tmp62 = fmul <2 x float> %tmp61, undef
  %tmp63 = fsub <2 x float> %tmp62, %tmp59
  %tmp64 = extractelement <2 x float> %tmp63, i64 0
  call void @eggs(float %tmp64) #2
  store <2 x float> %tmp3, <2 x float> addrspace(1)* undef, align 8
  store float 0.000000e+00, float addrspace(1)* undef, align 4
  ret void
}

declare float @barney() #2
declare void @eggs(float) #2
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { convergent nounwind "target-cpu"="gfx900" "target-features"="+fp32-denormals" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.cl", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(name: "var", arg: 8, scope: !4)
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, type: !12, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!5 = !DILocation(line: 69, scope: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
