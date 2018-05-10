; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -disable-output 2>&1 < %s | FileCheck %s -match-full-lines
;
; Derived from test-suite/MultiSource/Benchmarks/BitBench/uuencode/uuencode.c
;
; CHECK: remark: uuencode.c:75:18: The following errors keep this region from being a Scop.
; CHECK: remark: uuencode.c:83:3: Loop cannot be handled because it has multiple exits.
; CHECK: remark: uuencode.c:95:21: Invalid Scop candidate ends here.


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @encode_line(i8* nocapture readonly %input, i32 %offset, i32 %octets, i8* nocapture %line) !dbg !9 {
entry:
  br label %entry.split, !dbg !26

entry.split:                                      ; preds = %entry
  call void @llvm.dbg.value(metadata i8* %input, metadata !17, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 %offset, metadata !18, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %octets, metadata !19, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i8* %line, metadata !20, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !30
  %conv = trunc i32 %octets to i8, !dbg !31
  call void @llvm.dbg.value(metadata i8 %conv, metadata !32, metadata !DIExpression()), !dbg !37
  %tmp = and i8 %conv, 63, !dbg !39
  %addconv.i = add nuw nsw i8 %tmp, 32, !dbg !40
  store i8 %addconv.i, i8* %line, align 1, !dbg !41, !tbaa !42
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 %offset, metadata !18, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %octets, metadata !19, metadata !DIExpression()), !dbg !28
  %cmp220 = icmp sgt i32 %octets, 0, !dbg !45
  br i1 %cmp220, label %for.body.preheader, label %for.end, !dbg !46

for.body.preheader:                               ; preds = %entry.split
  %tmp1 = sext i32 %offset to i64, !dbg !47
  br label %for.body, !dbg !47

for.body:                                         ; preds = %if.end126, %for.body.preheader
  %indvars.iv = phi i64 [ %tmp1, %for.body.preheader ], [ %indvars.iv.next, %if.end126 ]
  %loffs.0223 = phi i32 [ 1, %for.body.preheader ], [ %inc49, %if.end126 ]
  %octets.addr.0221 = phi i32 [ %octets, %for.body.preheader ], [ %sub, %if.end126 ]
  call void @llvm.dbg.value(metadata i32 %loffs.0223, metadata !21, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !18, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %octets.addr.0221, metadata !19, metadata !DIExpression()), !dbg !28
  %cmp3 = icmp sgt i32 %octets.addr.0221, 2, !dbg !47
  br i1 %cmp3, label %if.end126, label %if.else, !dbg !49

if.else:                                          ; preds = %for.body
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !18, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %loffs.0223, metadata !21, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 %octets.addr.0221, metadata !19, metadata !DIExpression()), !dbg !28
  br label %for.end

if.then54:                                        ; No predecessors!
  %arrayidx56 = getelementptr inbounds i8, i8* %input, i64 %indvars.iv, !dbg !50
  %tmp2 = load i8, i8* %arrayidx56, align 1, !dbg !50, !tbaa !42
  %tmp3 = lshr i8 %tmp2, 2, !dbg !54
  call void @llvm.dbg.value(metadata i8 %tmp2, metadata !32, metadata !DIExpression(DW_OP_constu, 2, DW_OP_shra, DW_OP_stack_value)), !dbg !55
  %addconv.i210 = add nuw nsw i8 %tmp3, 32, !dbg !57
  call void @llvm.dbg.value(metadata i8 %addconv.i210, metadata !22, metadata !DIExpression()), !dbg !58
  %inc62 = add nuw nsw i32 %loffs.0223, 1, !dbg !59
  call void @llvm.dbg.value(metadata i32 %inc62, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp4 = zext i32 %loffs.0223 to i64, !dbg !60
  %arrayidx64 = getelementptr inbounds i8, i8* %line, i64 %tmp4, !dbg !60
  store i8 %addconv.i210, i8* %arrayidx64, align 1, !dbg !61, !tbaa !42
  %tmp5 = load i8, i8* %arrayidx56, align 1, !dbg !62, !tbaa !42
  %shl68 = shl i8 %tmp5, 4, !dbg !63
  call void @llvm.dbg.value(metadata i8 %shl68, metadata !32, metadata !DIExpression()), !dbg !64
  %tmp6 = and i8 %shl68, 48, !dbg !66
  %addconv.i208 = add nuw nsw i8 %tmp6, 32, !dbg !67
  call void @llvm.dbg.value(metadata i8 %addconv.i208, metadata !22, metadata !DIExpression()), !dbg !58
  %inc72 = add nuw nsw i32 %loffs.0223, 2, !dbg !68
  call void @llvm.dbg.value(metadata i32 %inc72, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp7 = zext i32 %inc62 to i64, !dbg !69
  %arrayidx74 = getelementptr inbounds i8, i8* %line, i64 %tmp7, !dbg !69
  store i8 %addconv.i208, i8* %arrayidx74, align 1, !dbg !70, !tbaa !42
  %inc75 = add nuw nsw i32 %loffs.0223, 3, !dbg !71
  call void @llvm.dbg.value(metadata i32 %inc75, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp8 = zext i32 %inc72 to i64, !dbg !72
  %arrayidx77 = getelementptr inbounds i8, i8* %line, i64 %tmp8, !dbg !72
  store i8 61, i8* %arrayidx77, align 1, !dbg !73, !tbaa !42
  %inc78 = add nuw nsw i32 %loffs.0223, 4, !dbg !74
  call void @llvm.dbg.value(metadata i32 %inc78, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp9 = zext i32 %inc75 to i64, !dbg !75
  %arrayidx80 = getelementptr inbounds i8, i8* %line, i64 %tmp9, !dbg !75
  store i8 61, i8* %arrayidx80, align 1, !dbg !76, !tbaa !42
  br label %for.end, !dbg !77

if.then84:                                        ; No predecessors!
  %arrayidx86 = getelementptr inbounds i8, i8* %input, i64 %indvars.iv, !dbg !78
  %tmp10 = load i8, i8* %arrayidx86, align 1, !dbg !78, !tbaa !42
  %tmp11 = lshr i8 %tmp10, 2, !dbg !82
  call void @llvm.dbg.value(metadata i8 %tmp10, metadata !32, metadata !DIExpression(DW_OP_constu, 2, DW_OP_shra, DW_OP_stack_value)), !dbg !83
  %addconv.i206 = add nuw nsw i8 %tmp11, 32, !dbg !85
  call void @llvm.dbg.value(metadata i8 %addconv.i206, metadata !22, metadata !DIExpression()), !dbg !58
  %inc92 = add nuw nsw i32 %loffs.0223, 1, !dbg !86
  call void @llvm.dbg.value(metadata i32 %inc92, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp12 = zext i32 %loffs.0223 to i64, !dbg !87
  %arrayidx94 = getelementptr inbounds i8, i8* %line, i64 %tmp12, !dbg !87
  store i8 %addconv.i206, i8* %arrayidx94, align 1, !dbg !88, !tbaa !42
  %tmp13 = load i8, i8* %arrayidx86, align 1, !dbg !89, !tbaa !42
  %shl98 = shl i8 %tmp13, 4, !dbg !90
  %tmp14 = add nsw i64 %indvars.iv, 1, !dbg !91
  %arrayidx101 = getelementptr inbounds i8, i8* %input, i64 %tmp14, !dbg !92
  %tmp15 = load i8, i8* %arrayidx101, align 1, !dbg !92, !tbaa !42
  %tmp16 = ashr i8 %tmp15, 4, !dbg !93
  %or104 = or i8 %tmp16, %shl98, !dbg !94
  call void @llvm.dbg.value(metadata i8 %or104, metadata !32, metadata !DIExpression()), !dbg !95
  %tmp17 = and i8 %or104, 63, !dbg !97
  %addconv.i204 = add nuw nsw i8 %tmp17, 32, !dbg !98
  call void @llvm.dbg.value(metadata i8 %addconv.i204, metadata !22, metadata !DIExpression()), !dbg !58
  %inc108 = add nuw nsw i32 %loffs.0223, 2, !dbg !99
  call void @llvm.dbg.value(metadata i32 %inc108, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp18 = zext i32 %inc92 to i64, !dbg !100
  %arrayidx110 = getelementptr inbounds i8, i8* %line, i64 %tmp18, !dbg !100
  store i8 %addconv.i204, i8* %arrayidx110, align 1, !dbg !101, !tbaa !42
  %tmp19 = load i8, i8* %arrayidx101, align 1, !dbg !102, !tbaa !42
  %shl115 = shl i8 %tmp19, 2, !dbg !103
  call void @llvm.dbg.value(metadata i8 %shl115, metadata !32, metadata !DIExpression()), !dbg !104
  %tmp20 = and i8 %shl115, 60, !dbg !106
  %addconv.i202 = add nuw nsw i8 %tmp20, 32, !dbg !107
  call void @llvm.dbg.value(metadata i8 %addconv.i202, metadata !22, metadata !DIExpression()), !dbg !58
  %inc119 = add nuw nsw i32 %loffs.0223, 3, !dbg !108
  call void @llvm.dbg.value(metadata i32 %inc119, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp21 = zext i32 %inc108 to i64, !dbg !109
  %arrayidx121 = getelementptr inbounds i8, i8* %line, i64 %tmp21, !dbg !109
  store i8 %addconv.i202, i8* %arrayidx121, align 1, !dbg !110, !tbaa !42
  %inc122 = add nuw nsw i32 %loffs.0223, 4, !dbg !111
  call void @llvm.dbg.value(metadata i32 %inc122, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp22 = zext i32 %inc119 to i64, !dbg !112
  %arrayidx124 = getelementptr inbounds i8, i8* %line, i64 %tmp22, !dbg !112
  store i8 61, i8* %arrayidx124, align 1, !dbg !113, !tbaa !42
  br label %for.end, !dbg !114

if.end126:                                        ; preds = %for.body
  %arrayidx6 = getelementptr inbounds i8, i8* %input, i64 %indvars.iv, !dbg !115
  %tmp23 = load i8, i8* %arrayidx6, align 1, !dbg !115, !tbaa !42
  %tmp24 = lshr i8 %tmp23, 2, !dbg !117
  call void @llvm.dbg.value(metadata i8 %tmp23, metadata !32, metadata !DIExpression(DW_OP_constu, 2, DW_OP_shra, DW_OP_stack_value)), !dbg !118
  %addconv.i218 = add nuw nsw i8 %tmp24, 32, !dbg !120
  call void @llvm.dbg.value(metadata i8 %addconv.i218, metadata !22, metadata !DIExpression()), !dbg !58
  %inc11 = add nuw nsw i32 %loffs.0223, 1, !dbg !121
  call void @llvm.dbg.value(metadata i32 %inc11, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp25 = zext i32 %loffs.0223 to i64, !dbg !122
  %arrayidx13 = getelementptr inbounds i8, i8* %line, i64 %tmp25, !dbg !122
  store i8 %addconv.i218, i8* %arrayidx13, align 1, !dbg !123, !tbaa !42
  %tmp26 = load i8, i8* %arrayidx6, align 1, !dbg !124, !tbaa !42
  %shl = shl i8 %tmp26, 4, !dbg !125
  %tmp27 = add nsw i64 %indvars.iv, 1, !dbg !126
  %arrayidx18 = getelementptr inbounds i8, i8* %input, i64 %tmp27, !dbg !127
  %tmp28 = load i8, i8* %arrayidx18, align 1, !dbg !127, !tbaa !42
  %tmp29 = ashr i8 %tmp28, 4, !dbg !128
  %or = or i8 %tmp29, %shl, !dbg !129
  call void @llvm.dbg.value(metadata i8 %or, metadata !32, metadata !DIExpression()), !dbg !130
  %tmp30 = and i8 %or, 63, !dbg !132
  %addconv.i216 = add nuw nsw i8 %tmp30, 32, !dbg !133
  call void @llvm.dbg.value(metadata i8 %addconv.i216, metadata !22, metadata !DIExpression()), !dbg !58
  %inc24 = add nuw nsw i32 %loffs.0223, 2, !dbg !134
  call void @llvm.dbg.value(metadata i32 %inc24, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp31 = zext i32 %inc11 to i64, !dbg !135
  %arrayidx26 = getelementptr inbounds i8, i8* %line, i64 %tmp31, !dbg !135
  store i8 %addconv.i216, i8* %arrayidx26, align 1, !dbg !136, !tbaa !42
  %tmp32 = load i8, i8* %arrayidx18, align 1, !dbg !137, !tbaa !42
  %shl31 = shl i8 %tmp32, 2, !dbg !138
  %tmp33 = add nsw i64 %indvars.iv, 2, !dbg !139
  %arrayidx34 = getelementptr inbounds i8, i8* %input, i64 %tmp33, !dbg !140
  %tmp34 = load i8, i8* %arrayidx34, align 1, !dbg !140, !tbaa !42
  %tmp35 = ashr i8 %tmp34, 6, !dbg !141
  %or37 = or i8 %tmp35, %shl31, !dbg !142
  call void @llvm.dbg.value(metadata i8 %or37, metadata !32, metadata !DIExpression()), !dbg !143
  %tmp36 = and i8 %or37, 63, !dbg !145
  %addconv.i214 = add nuw nsw i8 %tmp36, 32, !dbg !146
  call void @llvm.dbg.value(metadata i8 %addconv.i214, metadata !22, metadata !DIExpression()), !dbg !58
  %inc41 = add nuw nsw i32 %loffs.0223, 3, !dbg !147
  call void @llvm.dbg.value(metadata i32 %inc41, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp37 = zext i32 %inc24 to i64, !dbg !148
  %arrayidx43 = getelementptr inbounds i8, i8* %line, i64 %tmp37, !dbg !148
  store i8 %addconv.i214, i8* %arrayidx43, align 1, !dbg !149, !tbaa !42
  %tmp38 = load i8, i8* %arrayidx34, align 1, !dbg !150, !tbaa !42
  call void @llvm.dbg.value(metadata i8 %tmp38, metadata !32, metadata !DIExpression()), !dbg !151
  %tmp39 = and i8 %tmp38, 63, !dbg !153
  %addconv.i212 = add nuw nsw i8 %tmp39, 32, !dbg !154
  call void @llvm.dbg.value(metadata i8 %addconv.i212, metadata !22, metadata !DIExpression()), !dbg !58
  %inc49 = add nuw nsw i32 %loffs.0223, 4, !dbg !155
  call void @llvm.dbg.value(metadata i32 %inc49, metadata !21, metadata !DIExpression()), !dbg !30
  %tmp40 = zext i32 %inc41 to i64, !dbg !156
  %arrayidx51 = getelementptr inbounds i8, i8* %line, i64 %tmp40, !dbg !156
  store i8 %addconv.i212, i8* %arrayidx51, align 1, !dbg !157, !tbaa !42
  %indvars.iv.next = add nsw i64 %indvars.iv, 3, !dbg !158
  %sub = add nsw i32 %octets.addr.0221, -3, !dbg !159
  call void @llvm.dbg.value(metadata i32 %inc49, metadata !21, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 undef, metadata !18, metadata !DIExpression(DW_OP_plus_uconst, 3, DW_OP_stack_value)), !dbg !27
  call void @llvm.dbg.value(metadata i32 %sub, metadata !19, metadata !DIExpression()), !dbg !28
  %cmp = icmp eq i32 %octets.addr.0221, 3, !dbg !45
  br i1 %cmp, label %for.end, label %for.body, !dbg !46, !llvm.loop !160

for.end:                                          ; preds = %if.end126, %if.then84, %if.then54, %if.else, %entry.split
  %loffs.0.lcssa = phi i32 [ 1, %entry.split ], [ %loffs.0223, %if.else ], [ %inc122, %if.then84 ], [ %inc78, %if.then54 ], [ %inc49, %if.end126 ]
  call void @llvm.dbg.value(metadata i32 %loffs.0.lcssa, metadata !21, metadata !DIExpression()), !dbg !30
  %inc128 = add nsw i32 %loffs.0.lcssa, 1, !dbg !162
  call void @llvm.dbg.value(metadata i32 %inc128, metadata !21, metadata !DIExpression()), !dbg !30
  %idxprom129 = sext i32 %loffs.0.lcssa to i64, !dbg !163
  %arrayidx130 = getelementptr inbounds i8, i8* %line, i64 %idxprom129, !dbg !163
  store i8 10, i8* %arrayidx130, align 1, !dbg !164, !tbaa !42
  %idxprom131 = sext i32 %inc128 to i64, !dbg !165
  %arrayidx132 = getelementptr inbounds i8, i8* %line, i64 %idxprom131, !dbg !165
  store i8 0, i8* %arrayidx132, align 1, !dbg !166, !tbaa !42
  ret void, !dbg !167
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 330016) (llvm/trunk 330038)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "C:\5CUsers\5CMeinersbur\5Csrc\5Cllvm\5Ctools\5Cpolly\5Ctest\5Cuuencode.c", directory: "/tmp/runtest-kzqu096e")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 7.0.0 (trunk 330016) (llvm/trunk 330038)"}
!9 = distinct !DISubprogram(name: "encode_line", scope: !10, file: !10, line: 79, type: !11, isLocal: false, isDefinition: true, scopeLine: 79, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !16)
!10 = !DIFile(filename: "uuencode.c", directory: "/tmp/runtest-kzqu096e")
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !15, !15, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17, !18, !19, !20, !21, !22}
!17 = !DILocalVariable(name: "input", arg: 1, scope: !9, file: !10, line: 79, type: !13)
!18 = !DILocalVariable(name: "offset", arg: 2, scope: !9, file: !10, line: 79, type: !15)
!19 = !DILocalVariable(name: "octets", arg: 3, scope: !9, file: !10, line: 79, type: !15)
!20 = !DILocalVariable(name: "line", arg: 4, scope: !9, file: !10, line: 79, type: !13)
!21 = !DILocalVariable(name: "loffs", scope: !9, file: !10, line: 80, type: !15)
!22 = !DILocalVariable(name: "ch", scope: !23, file: !10, line: 86, type: !14)
!23 = distinct !DILexicalBlock(scope: !24, file: !10, line: 83, column: 55)
!24 = distinct !DILexicalBlock(scope: !25, file: !10, line: 83, column: 3)
!25 = distinct !DILexicalBlock(scope: !9, file: !10, line: 83, column: 3)
!26 = !DILocation(line: 79, column: 24, scope: !9)
!27 = !DILocation(line: 79, column: 35, scope: !9)
!28 = !DILocation(line: 79, column: 47, scope: !9)
!29 = !DILocation(line: 79, column: 61, scope: !9)
!30 = !DILocation(line: 80, column: 7, scope: !9)
!31 = !DILocation(line: 81, column: 27, scope: !9)
!32 = !DILocalVariable(name: "c", arg: 1, scope: !33, file: !10, line: 75, type: !14)
!33 = distinct !DISubprogram(name: "encode_char", scope: !10, file: !10, line: 75, type: !34, isLocal: false, isDefinition: true, scopeLine: 75, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !36)
!34 = !DISubroutineType(types: !35)
!35 = !{!15, !14}
!36 = !{!32}
!37 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !38)
!38 = distinct !DILocation(line: 81, column: 15, scope: !9)
!39 = !DILocation(line: 76, column: 17, scope: !33, inlinedAt: !38)
!40 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !38)
!41 = !DILocation(line: 81, column: 14, scope: !9)
!42 = !{!43, !43, i64 0}
!43 = !{!"omnipotent char", !44, i64 0}
!44 = !{!"Simple C/C++ TBAA"}
!45 = !DILocation(line: 83, column: 24, scope: !24)
!46 = !DILocation(line: 83, column: 3, scope: !25)
!47 = !DILocation(line: 87, column: 16, scope: !48)
!48 = distinct !DILexicalBlock(scope: !23, file: !10, line: 87, column: 9)
!49 = !DILocation(line: 87, column: 9, scope: !23)
!50 = !DILocation(line: 99, column: 23, scope: !51)
!51 = distinct !DILexicalBlock(scope: !52, file: !10, line: 98, column: 24)
!52 = distinct !DILexicalBlock(scope: !53, file: !10, line: 98, column: 11)
!53 = distinct !DILexicalBlock(scope: !48, file: !10, line: 97, column: 12)
!54 = !DILocation(line: 99, column: 37, scope: !51)
!55 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !56)
!56 = distinct !DILocation(line: 99, column: 10, scope: !51)
!57 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !56)
!58 = !DILocation(line: 86, column: 10, scope: !23)
!59 = !DILocation(line: 100, column: 15, scope: !51)
!60 = !DILocation(line: 100, column: 5, scope: !51)
!61 = !DILocation(line: 100, column: 19, scope: !51)
!62 = !DILocation(line: 101, column: 23, scope: !51)
!63 = !DILocation(line: 101, column: 37, scope: !51)
!64 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !65)
!65 = distinct !DILocation(line: 101, column: 10, scope: !51)
!66 = !DILocation(line: 76, column: 17, scope: !33, inlinedAt: !65)
!67 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !65)
!68 = !DILocation(line: 102, column: 15, scope: !51)
!69 = !DILocation(line: 102, column: 5, scope: !51)
!70 = !DILocation(line: 102, column: 19, scope: !51)
!71 = !DILocation(line: 103, column: 15, scope: !51)
!72 = !DILocation(line: 103, column: 5, scope: !51)
!73 = !DILocation(line: 103, column: 19, scope: !51)
!74 = !DILocation(line: 104, column: 15, scope: !51)
!75 = !DILocation(line: 104, column: 5, scope: !51)
!76 = !DILocation(line: 104, column: 19, scope: !51)
!77 = !DILocation(line: 106, column: 4, scope: !51)
!78 = !DILocation(line: 108, column: 25, scope: !79)
!79 = distinct !DILexicalBlock(scope: !80, file: !10, line: 107, column: 22)
!80 = distinct !DILexicalBlock(scope: !81, file: !10, line: 107, column: 9)
!81 = distinct !DILexicalBlock(scope: !52, file: !10, line: 106, column: 11)
!82 = !DILocation(line: 108, column: 40, scope: !79)
!83 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !84)
!84 = distinct !DILocation(line: 108, column: 11, scope: !79)
!85 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !84)
!86 = !DILocation(line: 109, column: 16, scope: !79)
!87 = !DILocation(line: 109, column: 6, scope: !79)
!88 = !DILocation(line: 109, column: 20, scope: !79)
!89 = !DILocation(line: 110, column: 25, scope: !79)
!90 = !DILocation(line: 110, column: 39, scope: !79)
!91 = !DILocation(line: 110, column: 60, scope: !79)
!92 = !DILocation(line: 110, column: 48, scope: !79)
!93 = !DILocation(line: 110, column: 64, scope: !79)
!94 = !DILocation(line: 110, column: 45, scope: !79)
!95 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !96)
!96 = distinct !DILocation(line: 110, column: 11, scope: !79)
!97 = !DILocation(line: 76, column: 17, scope: !33, inlinedAt: !96)
!98 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !96)
!99 = !DILocation(line: 111, column: 16, scope: !79)
!100 = !DILocation(line: 111, column: 6, scope: !79)
!101 = !DILocation(line: 111, column: 20, scope: !79)
!102 = !DILocation(line: 112, column: 24, scope: !79)
!103 = !DILocation(line: 112, column: 40, scope: !79)
!104 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !105)
!105 = distinct !DILocation(line: 112, column: 11, scope: !79)
!106 = !DILocation(line: 76, column: 17, scope: !33, inlinedAt: !105)
!107 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !105)
!108 = !DILocation(line: 113, column: 16, scope: !79)
!109 = !DILocation(line: 113, column: 6, scope: !79)
!110 = !DILocation(line: 113, column: 20, scope: !79)
!111 = !DILocation(line: 114, column: 16, scope: !79)
!112 = !DILocation(line: 114, column: 6, scope: !79)
!113 = !DILocation(line: 114, column: 20, scope: !79)
!114 = !DILocation(line: 116, column: 5, scope: !79)
!115 = !DILocation(line: 88, column: 25, scope: !116)
!116 = distinct !DILexicalBlock(scope: !48, file: !10, line: 87, column: 22)
!117 = !DILocation(line: 88, column: 39, scope: !116)
!118 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !119)
!119 = distinct !DILocation(line: 88, column: 12, scope: !116)
!120 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !119)
!121 = !DILocation(line: 89, column: 17, scope: !116)
!122 = !DILocation(line: 89, column: 7, scope: !116)
!123 = !DILocation(line: 89, column: 21, scope: !116)
!124 = !DILocation(line: 90, column: 26, scope: !116)
!125 = !DILocation(line: 90, column: 40, scope: !116)
!126 = !DILocation(line: 90, column: 61, scope: !116)
!127 = !DILocation(line: 90, column: 49, scope: !116)
!128 = !DILocation(line: 90, column: 65, scope: !116)
!129 = !DILocation(line: 90, column: 46, scope: !116)
!130 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !131)
!131 = distinct !DILocation(line: 90, column: 12, scope: !116)
!132 = !DILocation(line: 76, column: 17, scope: !33, inlinedAt: !131)
!133 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !131)
!134 = !DILocation(line: 91, column: 17, scope: !116)
!135 = !DILocation(line: 91, column: 7, scope: !116)
!136 = !DILocation(line: 91, column: 21, scope: !116)
!137 = !DILocation(line: 92, column: 26, scope: !116)
!138 = !DILocation(line: 92, column: 42, scope: !116)
!139 = !DILocation(line: 92, column: 63, scope: !116)
!140 = !DILocation(line: 92, column: 51, scope: !116)
!141 = !DILocation(line: 92, column: 67, scope: !116)
!142 = !DILocation(line: 92, column: 48, scope: !116)
!143 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !144)
!144 = distinct !DILocation(line: 92, column: 12, scope: !116)
!145 = !DILocation(line: 76, column: 17, scope: !33, inlinedAt: !144)
!146 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !144)
!147 = !DILocation(line: 93, column: 17, scope: !116)
!148 = !DILocation(line: 93, column: 7, scope: !116)
!149 = !DILocation(line: 93, column: 21, scope: !116)
!150 = !DILocation(line: 94, column: 25, scope: !116)
!151 = !DILocation(line: 75, column: 18, scope: !33, inlinedAt: !152)
!152 = distinct !DILocation(line: 94, column: 12, scope: !116)
!153 = !DILocation(line: 76, column: 17, scope: !33, inlinedAt: !152)
!154 = !DILocation(line: 76, column: 13, scope: !33, inlinedAt: !152)
!155 = !DILocation(line: 95, column: 17, scope: !116)
!156 = !DILocation(line: 95, column: 7, scope: !116)
!157 = !DILocation(line: 95, column: 21, scope: !116)
!158 = !DILocation(line: 83, column: 36, scope: !24)
!159 = !DILocation(line: 83, column: 49, scope: !24)
!160 = distinct !{!160, !46, !161}
!161 = !DILocation(line: 119, column: 3, scope: !25)
!162 = !DILocation(line: 121, column: 13, scope: !9)
!163 = !DILocation(line: 121, column: 3, scope: !9)
!164 = !DILocation(line: 121, column: 17, scope: !9)
!165 = !DILocation(line: 122, column: 3, scope: !9)
!166 = !DILocation(line: 122, column: 15, scope: !9)
!167 = !DILocation(line: 124, column: 1, scope: !9)
