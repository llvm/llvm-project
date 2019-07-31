; RUN: opt -S -loop-fusion -pass-remarks-missed=loop-fusion -disable-output < %s 2>&1 | FileCheck %s
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@B = common global [1024 x i32] zeroinitializer, align 16, !dbg !0

; CHECK: remark: diagnostics_missed.c:18:3: [non_adjacent]: entry and for.end: Loops are not adjacent
define void @non_adjacent(i32* noalias %A) !dbg !67 {
entry:
    br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
    %exitcond1 = icmp ne i64 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  %sub = add nsw i64 %i.0, -3
  %add = add nuw nsw i64 %i.0, 3
  %mul = mul nsw i64 %sub, %add
  %rem = srem i64 %mul, %i.0
  %conv = trunc i64 %rem to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.0
  store i32 %conv, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1, !dbg !86
  br label %for.cond, !dbg !87, !llvm.loop !88

for.end:                                          ; preds = %for.cond.cleanup
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc13, %for.end
  %i1.0 = phi i64 [ 0, %for.end ], [ %inc14, %for.inc13 ]
  %exitcond = icmp ne i64 %i1.0, 100
  br i1 %exitcond, label %for.body6, label %for.cond.cleanup5

for.cond.cleanup5:                                ; preds = %for.cond2
  br label %for.end15

for.body6:                                        ; preds = %for.cond2
  %sub7 = add nsw i64 %i1.0, -3
  %add8 = add nuw nsw i64 %i1.0, 3
  %mul9 = mul nsw i64 %sub7, %add8
  %rem10 = srem i64 %mul9, %i1.0
  %conv11 = trunc i64 %rem10 to i32
  %arrayidx12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %i1.0
  store i32 %conv11, i32* %arrayidx12, align 4
  br label %for.inc13

for.inc13:                                        ; preds = %for.body6
  %inc14 = add nuw nsw i64 %i1.0, 1, !dbg !100
  br label %for.cond2, !dbg !101, !llvm.loop !102

for.end15:                                        ; preds = %for.cond.cleanup5
  ret void
}


; CHECK: remark: diagnostics_missed.c:28:3: [different_bounds]: entry and for.end: Loop trip counts are not the same
define void @different_bounds(i32* noalias %A) !dbg !105 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond1 = icmp ne i64 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  %sub = add nsw i64 %i.0, -3
  %add = add nuw nsw i64 %i.0, 3
  %mul = mul nsw i64 %sub, %add
  %rem = srem i64 %mul, %i.0
  %conv = trunc i64 %rem to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.0
  store i32 %conv, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1, !dbg !123
  br label %for.cond, !dbg !124, !llvm.loop !125

for.end:                                          ; preds = %for.cond.cleanup
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc13, %for.end
  %i1.0 = phi i64 [ 0, %for.end ], [ %inc14, %for.inc13 ]
  %exitcond = icmp ne i64 %i1.0, 200
  br i1 %exitcond, label %for.body6, label %for.cond.cleanup5

for.cond.cleanup5:                                ; preds = %for.cond2
  br label %for.end15

for.body6:                                        ; preds = %for.cond2
  %sub7 = add nsw i64 %i1.0, -3
  %add8 = add nuw nsw i64 %i1.0, 3
  %mul9 = mul nsw i64 %sub7, %add8
  %rem10 = srem i64 %mul9, %i1.0
  %conv11 = trunc i64 %rem10 to i32
  %arrayidx12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %i1.0
  store i32 %conv11, i32* %arrayidx12, align 4
  br label %for.inc13

for.inc13:                                        ; preds = %for.body6
  %inc14 = add nuw nsw i64 %i1.0, 1
  br label %for.cond2, !dbg !138, !llvm.loop !139

for.end15:                                        ; preds = %for.cond.cleanup5
  ret void
}

; CHECK: remark: diagnostics_missed.c:38:3: [negative_dependence]: entry and for.end: Loop has a non-empty preheader
define void @negative_dependence(i32* noalias %A) !dbg !142 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ 0, %entry ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 100
  br i1 %exitcond3, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  %tmp = trunc i64 %indvars.iv1 to i32
  store i32 %tmp, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond, !dbg !160, !llvm.loop !161

for.end:                                          ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 0, metadata !147, metadata !DIExpression()), !dbg !163
  br label %for.cond2, !dbg !164

for.cond2:                                        ; preds = %for.inc10, %for.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc10 ], [ 0, %for.end ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body5, label %for.end12

for.body5:                                        ; preds = %for.cond2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %tmp4 = load i32, i32* %arrayidx7, align 4
  %mul = shl nsw i32 %tmp4, 1
  %arrayidx9 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx9, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.body5
  br label %for.cond2

for.end12:                                        ; preds = %for.cond.
  ret void, !dbg !178
}

; CHECK: remark: diagnostics_missed.c:51:3: [sumTest]: entry and for.cond2.preheader: Dependencies prevent fusion
define i32 @sumTest(i32* noalias %A) !dbg !179 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ 0, %entry ]
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 100
  br i1 %exitcond3, label %for.body, label %for.cond2

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %tmp
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond, !dbg !199, !llvm.loop !200

for.cond2:                                        ; preds = %for.inc10, %for.cond
  %sum.0.lcssa = phi i32 [ %sum.0, %for.cond ], [ %sum.0.lcssa, %for.inc10 ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc10 ], [ 0, %for.cond ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body5, label %for.end12

for.body5:                                        ; preds = %for.cond2
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp4 = load i32, i32* %arrayidx7, align 4
  %div = sdiv i32 %tmp4, %sum.0.lcssa
  %arrayidx9 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %div, i32* %arrayidx9, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.body5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond2

for.end12:                                        ; preds = %for.cond2
  ret i32 %sum.0.lcssa, !dbg !215
}

declare void @llvm.dbg.value(metadata, metadata, metadata)


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "B", scope: !2, file: !6, line: 46, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (git@github.ibm.com:compiler/llvm-project.git 23c4baaa9f5b33d2d52eda981d376c6b0a7a3180)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "diagnostics_missed.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "diagnostics_missed.c", directory: "/tmp")
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 32768, elements: !9)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DISubrange(count: 1024)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 7, !"PIC Level", i32 2}
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !20)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!67 = distinct !DISubprogram(name: "non_adjacent", scope: !6, file: !6, line: 17, type: !17, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !68)
!68 = !{!69, !70, !73}
!69 = !DILocalVariable(name: "A", arg: 1, scope: !67, file: !6, line: 17, type: !19)
!70 = !DILocalVariable(name: "i", scope: !71, file: !6, line: 18, type: !72)
!71 = distinct !DILexicalBlock(scope: !67, file: !6, line: 18, column: 3)
!72 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!73 = !DILocalVariable(name: "i", scope: !74, file: !6, line: 22, type: !72)
!74 = distinct !DILexicalBlock(scope: !67, file: !6, line: 22, column: 3)
!79 = distinct !DILexicalBlock(scope: !71, file: !6, line: 18, column: 3)
!80 = !DILocation(line: 18, column: 3, scope: !71)
!86 = !DILocation(line: 18, column: 30, scope: !79)
!87 = !DILocation(line: 18, column: 3, scope: !79)
!88 = distinct !{!88, !80, !89}
!89 = !DILocation(line: 20, column: 3, scope: !71)
!93 = distinct !DILexicalBlock(scope: !74, file: !6, line: 22, column: 3)
!94 = !DILocation(line: 22, column: 3, scope: !74)
!100 = !DILocation(line: 22, column: 30, scope: !93)
!101 = !DILocation(line: 22, column: 3, scope: !93)
!102 = distinct !{!102, !94, !103}
!103 = !DILocation(line: 24, column: 3, scope: !74)
!105 = distinct !DISubprogram(name: "different_bounds", scope: !6, file: !6, line: 27, type: !17, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !106)
!106 = !{!107, !108, !110}
!107 = !DILocalVariable(name: "A", arg: 1, scope: !105, file: !6, line: 27, type: !19)
!108 = !DILocalVariable(name: "i", scope: !109, file: !6, line: 28, type: !72)
!109 = distinct !DILexicalBlock(scope: !105, file: !6, line: 28, column: 3)
!110 = !DILocalVariable(name: "i", scope: !111, file: !6, line: 32, type: !72)
!111 = distinct !DILexicalBlock(scope: !105, file: !6, line: 32, column: 3)
!116 = distinct !DILexicalBlock(scope: !109, file: !6, line: 28, column: 3)
!117 = !DILocation(line: 28, column: 3, scope: !109)
!123 = !DILocation(line: 28, column: 30, scope: !116)
!124 = !DILocation(line: 28, column: 3, scope: !116)
!125 = distinct !{!125, !117, !126}
!126 = !DILocation(line: 30, column: 3, scope: !109)
!130 = distinct !DILexicalBlock(scope: !111, file: !6, line: 32, column: 3)
!131 = !DILocation(line: 32, column: 3, scope: !111)
!138 = !DILocation(line: 32, column: 3, scope: !130)
!139 = distinct !{!139, !131, !140}
!140 = !DILocation(line: 34, column: 3, scope: !111)
!142 = distinct !DISubprogram(name: "negative_dependence", scope: !6, file: !6, line: 37, type: !17, scopeLine: 37, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !143)
!143 = !{!144, !145, !147}
!144 = !DILocalVariable(name: "A", arg: 1, scope: !142, file: !6, line: 37, type: !19)
!145 = !DILocalVariable(name: "i", scope: !146, file: !6, line: 38, type: !8)
!146 = distinct !DILexicalBlock(scope: !142, file: !6, line: 38, column: 3)
!147 = !DILocalVariable(name: "i", scope: !148, file: !6, line: 42, type: !8)
!148 = distinct !DILexicalBlock(scope: !142, file: !6, line: 42, column: 3)
!153 = distinct !DILexicalBlock(scope: !146, file: !6, line: 38, column: 3)
!154 = !DILocation(line: 38, column: 3, scope: !146)
!160 = !DILocation(line: 38, column: 3, scope: !153)
!161 = distinct !{!161, !154, !162}
!162 = !DILocation(line: 40, column: 3, scope: !146)
!163 = !DILocation(line: 0, scope: !148)
!164 = !DILocation(line: 42, column: 8, scope: !148)
!178 = !DILocation(line: 45, column: 1, scope: !142)
!179 = distinct !DISubprogram(name: "sumTest", scope: !6, file: !6, line: 48, type: !180, scopeLine: 48, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !182)
!180 = !DISubroutineType(types: !181)
!181 = !{!8, !19}
!182 = !{!183, !184, !185, !187}
!183 = !DILocalVariable(name: "A", arg: 1, scope: !179, file: !6, line: 48, type: !19)
!184 = !DILocalVariable(name: "sum", scope: !179, file: !6, line: 49, type: !8)
!185 = !DILocalVariable(name: "i", scope: !186, file: !6, line: 51, type: !8)
!186 = distinct !DILexicalBlock(scope: !179, file: !6, line: 51, column: 3)
!187 = !DILocalVariable(name: "i", scope: !188, file: !6, line: 54, type: !8)
!188 = distinct !DILexicalBlock(scope: !179, file: !6, line: 54, column: 3)
!193 = distinct !DILexicalBlock(scope: !186, file: !6, line: 51, column: 3)
!194 = !DILocation(line: 51, column: 3, scope: !186)
!199 = !DILocation(line: 51, column: 3, scope: !193)
!200 = distinct !{!200, !194, !201}
!201 = !DILocation(line: 52, column: 15, scope: !186)
!215 = !DILocation(line: 57, column: 3, scope: !179)
