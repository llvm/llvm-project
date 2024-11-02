; RUN: opt -S -passes=loop-fusion -pass-remarks-missed=loop-fusion -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@B = common global [1024 x i32] zeroinitializer, align 16, !dbg !0

; CHECK: remark: diagnostics_missed.c:18:3: [non_adjacent]: entry and for.end: Loops are not adjacent
define void @non_adjacent(i32* noalias %A) !dbg !14 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  br label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %sub = add nsw i64 %i.02, -3
  %add = add nuw nsw i64 %i.02, 3
  %mul = mul nsw i64 %sub, %add
  %rem = srem i64 %mul, %i.02
  %conv = trunc i64 %rem to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.02
  store i32 %conv, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.02, 1, !dbg !26
  %exitcond1 = icmp ne i64 %inc, 100
  br i1 %exitcond1, label %for.body, label %for.cond.cleanup, !llvm.loop !28

for.end:                                          ; preds = %for.cond.cleanup
  br label %for.body6

for.cond.cleanup5:                                ; preds = %for.inc13
  br label %for.end15

for.body6:                                        ; preds = %for.end, %for.inc13
  %i1.01 = phi i64 [ 0, %for.end ], [ %inc14, %for.inc13 ]
  %sub7 = add nsw i64 %i1.01, -3
  %add8 = add nuw nsw i64 %i1.01, 3
  %mul9 = mul nsw i64 %sub7, %add8
  %rem10 = srem i64 %mul9, %i1.01
  %conv11 = trunc i64 %rem10 to i32
  %arrayidx12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %i1.01
  store i32 %conv11, i32* %arrayidx12, align 4
  br label %for.inc13

for.inc13:                                        ; preds = %for.body6
  %inc14 = add nuw nsw i64 %i1.01, 1, !dbg !31
  %exitcond = icmp ne i64 %inc14, 100
  br i1 %exitcond, label %for.body6, label %for.cond.cleanup5, !llvm.loop !33

for.end15:                                        ; preds = %for.cond.cleanup5
  ret void
}

; CHECK: remark: diagnostics_missed.c:28:3: [different_bounds]: entry and for.end: Loop trip counts are not the same
define void @different_bounds(i32* noalias %A) !dbg !36 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  br label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %sub = add nsw i64 %i.02, -3
  %add = add nuw nsw i64 %i.02, 3
  %mul = mul nsw i64 %sub, %add
  %rem = srem i64 %mul, %i.02
  %conv = trunc i64 %rem to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.02
  store i32 %conv, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.02, 1, !dbg !43
  %exitcond1 = icmp ne i64 %inc, 100
  br i1 %exitcond1, label %for.body, label %for.cond.cleanup, !llvm.loop !45

for.end:                                          ; preds = %for.cond.cleanup
  br label %for.body6

for.cond.cleanup5:                                ; preds = %for.inc13
  br label %for.end15

for.body6:                                        ; preds = %for.end, %for.inc13
  %i1.01 = phi i64 [ 0, %for.end ], [ %inc14, %for.inc13 ]
  %sub7 = add nsw i64 %i1.01, -3
  %add8 = add nuw nsw i64 %i1.01, 3
  %mul9 = mul nsw i64 %sub7, %add8
  %rem10 = srem i64 %mul9, %i1.01
  %conv11 = trunc i64 %rem10 to i32
  %arrayidx12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %i1.01
  store i32 %conv11, i32* %arrayidx12, align 4
  br label %for.inc13

for.inc13:                                        ; preds = %for.body6
  %inc14 = add nuw nsw i64 %i1.01, 1
  %exitcond = icmp ne i64 %inc14, 200
  br i1 %exitcond, label %for.body6, label %for.cond.cleanup5, !llvm.loop !48

for.end15:                                        ; preds = %for.cond.cleanup5
  ret void
}

; CHECK: remark: diagnostics_missed.c:38:3: [negative_dependence]: entry and for.end: Dependencies prevent fusion
define void @negative_dependence(i32* noalias %A) !dbg !51 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv13 = phi i64 [ 0, %entry ], [ %indvars.iv.next2, %for.inc ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv13
  %tmp = trunc i64 %indvars.iv13 to i32
  store i32 %tmp, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv13, 1
  %exitcond3 = icmp ne i64 %indvars.iv.next2, 100
  br i1 %exitcond3, label %for.body, label %for.end, !llvm.loop !58

for.end:                                          ; preds = %for.inc
  call void @llvm.dbg.value(metadata i32 0, metadata !56, metadata !DIExpression()), !dbg !61
  br label %for.body5

for.body5:                                        ; preds = %for.end, %for.inc10
  %indvars.iv2 = phi i64 [ 0, %for.end ], [ %indvars.iv.next, %for.inc10 ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %tmp4 = load i32, i32* %arrayidx7, align 4
  %mul = shl nsw i32 %tmp4, 1
  %arrayidx9 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv2
  store i32 %mul, i32* %arrayidx9, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.body5
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.body5, label %for.end12

for.end12:                                        ; preds = %for.inc10
  ret void, !dbg !62
}

; CHECK: remark: diagnostics_missed.c:51:3: [sumTest]: entry and for.cond2.preheader: Dependencies prevent fusion
define i32 @sumTest(i32* noalias %A) !dbg !63 {
entry:
  br label %for.body

for.cond2.preheader:                              ; preds = %for.inc
  br label %for.body5

for.body:                                         ; preds = %entry, %for.inc
  %sum.04 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %indvars.iv13 = phi i64 [ 0, %entry ], [ %indvars.iv.next2, %for.inc ]
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv13
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.04, %tmp
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv13, 1
  %exitcond3 = icmp ne i64 %indvars.iv.next2, 100
  br i1 %exitcond3, label %for.body, label %for.cond2.preheader, !llvm.loop !73

for.body5:                                        ; preds = %for.cond2.preheader, %for.inc10
  %indvars.iv2 = phi i64 [ 0, %for.cond2.preheader ], [ %indvars.iv.next, %for.inc10 ]
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv2
  %tmp4 = load i32, i32* %arrayidx7, align 4
  %div = sdiv i32 %tmp4, %add
  %arrayidx9 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvars.iv2
  store i32 %div, i32* %arrayidx9, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.body5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.body5, label %for.end12

for.end12:                                        ; preds = %for.inc10
  ret i32 %add, !dbg !76
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; CHECK: remark: diagnostics_missed.c:62:3: [unsafe_preheader]: for.first.preheader and for.second.preheader: Loop has a non-empty preheader with instructions that cannot be moved
define void @unsafe_preheader(i32* noalias %A, i32* noalias %B) {
for.first.preheader:
  br label %for.first, !dbg !80

for.first:
  %i.02 = phi i64 [ 0, %for.first.preheader ], [ %inc, %for.first ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.02
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.02, 1
  %cmp = icmp slt i64 %inc, 100
  br i1 %cmp, label %for.first, label %for.second.preheader

for.second.preheader:
  call void @bar()
  br label %for.second

for.second:
  %j.01 = phi i64 [ 0, %for.second.preheader ], [ %inc6, %for.second ]
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %j.01
  store i32 0, i32* %arrayidx4, align 4
  %inc6 = add nsw i64 %j.01, 1
  %cmp2 = icmp slt i64 %inc6, 100
  br i1 %cmp2, label %for.second, label %for.end

for.end:
  ret void
}

; CHECK: remark: diagnostics_missed.c:67:3: [unsafe_exitblock]: for.first.preheader and for.second.preheader: Candidate has a non-empty exit block with instructions that cannot be moved
define void @unsafe_exitblock(i32* noalias %A, i32* noalias %B, i64 %N) {
for.first.guard:
  %cmp3 = icmp slt i64 0, %N
  br i1 %cmp3, label %for.first.preheader, label %for.second.guard

for.first.preheader:
  br label %for.first, !dbg !83

for.first:
  %i.04 = phi i64 [ %inc, %for.first ], [ 0, %for.first.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.04
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.04, 1
  %cmp = icmp slt i64 %inc, %N
  br i1 %cmp, label %for.first, label %for.first.exit

for.first.exit:
  call void @bar()
  br label %for.second.guard

for.second.guard:
  %cmp21 = icmp slt i64 0, %N
  br i1 %cmp21, label %for.second.preheader, label %for.end

for.second.preheader:
  br label %for.second

for.second:
  %j.02 = phi i64 [ %inc6, %for.second ], [ 0, %for.second.preheader ]
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %j.02
  store i32 0, i32* %arrayidx4, align 4
  %inc6 = add nsw i64 %j.02, 1
  %cmp2 = icmp slt i64 %inc6, %N
  br i1 %cmp2, label %for.second, label %for.second.exit

for.second.exit:
  br label %for.end

for.end:
  ret void
}

; CHECK: remark: diagnostics_missed.c:72:3: [unsafe_guardblock]: for.first.preheader and for.second.preheader: Candidate has a non-empty guard block with instructions that cannot be moved
define void @unsafe_guardblock(i32* noalias %A, i32* noalias %B, i64 %N) {
for.first.guard:
  %cmp3 = icmp slt i64 0, %N
  br i1 %cmp3, label %for.first.preheader, label %for.second.guard

for.first.preheader:
  br label %for.first, !dbg !86

for.first:
  %i.04 = phi i64 [ %inc, %for.first ], [ 0, %for.first.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.04
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.04, 1
  %cmp = icmp slt i64 %inc, %N
  br i1 %cmp, label %for.first, label %for.first.exit

for.first.exit:
  br label %for.second.guard

for.second.guard:
  call void @bar()
  %cmp21 = icmp slt i64 0, %N
  br i1 %cmp21, label %for.second.preheader, label %for.end

for.second.preheader:
  br label %for.second

for.second:
  %j.02 = phi i64 [ %inc6, %for.second ], [ 0, %for.second.preheader ]
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %j.02
  store i32 0, i32* %arrayidx4, align 4
  %inc6 = add nsw i64 %j.02, 1
  %cmp2 = icmp slt i64 %inc6, %N
  br i1 %cmp2, label %for.second, label %for.second.exit

for.second.exit:
  br label %for.end

for.end:
  ret void
}

declare void @bar()

attributes #0 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "B", scope: !2, file: !3, line: 46, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (git@github.ibm.com:compiler/llvm-project.git 23c4baaa9f5b33d2d52eda981d376c6b0a7a3180)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "diagnostics_missed.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32768, elements: !8)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 1024)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = distinct !DISubprogram(name: "non_adjacent", scope: !3, file: !3, line: 17, type: !15, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!19 = !{!20, !21, !24}
!20 = !DILocalVariable(name: "A", arg: 1, scope: !14, file: !3, line: 17, type: !17)
!21 = !DILocalVariable(name: "i", scope: !22, file: !3, line: 18, type: !23)
!22 = distinct !DILexicalBlock(scope: !14, file: !3, line: 18, column: 3)
!23 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!24 = !DILocalVariable(name: "i", scope: !25, file: !3, line: 22, type: !23)
!25 = distinct !DILexicalBlock(scope: !14, file: !3, line: 22, column: 3)
!26 = !DILocation(line: 18, column: 30, scope: !27)
!27 = distinct !DILexicalBlock(scope: !22, file: !3, line: 18, column: 3)
!28 = distinct !{!28, !29, !30}
!29 = !DILocation(line: 18, column: 3, scope: !22)
!30 = !DILocation(line: 20, column: 3, scope: !22)
!31 = !DILocation(line: 22, column: 30, scope: !32)
!32 = distinct !DILexicalBlock(scope: !25, file: !3, line: 22, column: 3)
!33 = distinct !{!33, !34, !35}
!34 = !DILocation(line: 22, column: 3, scope: !25)
!35 = !DILocation(line: 24, column: 3, scope: !25)
!36 = distinct !DISubprogram(name: "different_bounds", scope: !3, file: !3, line: 27, type: !15, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !37)
!37 = !{!38, !39, !41}
!38 = !DILocalVariable(name: "A", arg: 1, scope: !36, file: !3, line: 27, type: !17)
!39 = !DILocalVariable(name: "i", scope: !40, file: !3, line: 28, type: !23)
!40 = distinct !DILexicalBlock(scope: !36, file: !3, line: 28, column: 3)
!41 = !DILocalVariable(name: "i", scope: !42, file: !3, line: 32, type: !23)
!42 = distinct !DILexicalBlock(scope: !36, file: !3, line: 32, column: 3)
!43 = !DILocation(line: 28, column: 30, scope: !44)
!44 = distinct !DILexicalBlock(scope: !40, file: !3, line: 28, column: 3)
!45 = distinct !{!45, !46, !47}
!46 = !DILocation(line: 28, column: 3, scope: !40)
!47 = !DILocation(line: 30, column: 3, scope: !40)
!48 = distinct !{!48, !49, !50}
!49 = !DILocation(line: 32, column: 3, scope: !42)
!50 = !DILocation(line: 34, column: 3, scope: !42)
!51 = distinct !DISubprogram(name: "negative_dependence", scope: !3, file: !3, line: 37, type: !15, scopeLine: 37, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !52)
!52 = !{!53, !54, !56}
!53 = !DILocalVariable(name: "A", arg: 1, scope: !51, file: !3, line: 37, type: !17)
!54 = !DILocalVariable(name: "i", scope: !55, file: !3, line: 38, type: !7)
!55 = distinct !DILexicalBlock(scope: !51, file: !3, line: 38, column: 3)
!56 = !DILocalVariable(name: "i", scope: !57, file: !3, line: 42, type: !7)
!57 = distinct !DILexicalBlock(scope: !51, file: !3, line: 42, column: 3)
!58 = distinct !{!58, !59, !60}
!59 = !DILocation(line: 38, column: 3, scope: !55)
!60 = !DILocation(line: 40, column: 3, scope: !55)
!61 = !DILocation(line: 0, scope: !57)
!62 = !DILocation(line: 45, column: 1, scope: !51)
!63 = distinct !DISubprogram(name: "sumTest", scope: !3, file: !3, line: 48, type: !64, scopeLine: 48, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !66)
!64 = !DISubroutineType(types: !65)
!65 = !{!7, !17}
!66 = !{!67, !68, !69, !71}
!67 = !DILocalVariable(name: "A", arg: 1, scope: !63, file: !3, line: 48, type: !17)
!68 = !DILocalVariable(name: "sum", scope: !63, file: !3, line: 49, type: !7)
!69 = !DILocalVariable(name: "i", scope: !70, file: !3, line: 51, type: !7)
!70 = distinct !DILexicalBlock(scope: !63, file: !3, line: 51, column: 3)
!71 = !DILocalVariable(name: "i", scope: !72, file: !3, line: 54, type: !7)
!72 = distinct !DILexicalBlock(scope: !63, file: !3, line: 54, column: 3)
!73 = distinct !{!73, !74, !75}
!74 = !DILocation(line: 51, column: 3, scope: !70)
!75 = !DILocation(line: 52, column: 15, scope: !70)
!76 = !DILocation(line: 57, column: 3, scope: !63)
!77 = distinct !DISubprogram(name: "unsafe_preheader", scope: !3, file: !3, line: 60, type: !15, scopeLine: 60, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
!78 = !{}
!79 = distinct !DILexicalBlock(scope: !77, file: !3, line: 3, column: 5)
!80 = !DILocation(line: 62, column: 3, scope: !79)
!81 = distinct !DISubprogram(name: "unsafe_exitblock", scope: !3, file: !3, line: 65, type: !15, scopeLine: 60, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
!82 = distinct !DILexicalBlock(scope: !81, file: !3, line: 3, column: 5)
!83 = !DILocation(line: 67, column: 3, scope: !82)
!84 = distinct !DISubprogram(name: "unsafe_guardblock", scope: !3, file: !3, line: 70, type: !15, scopeLine: 60, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
!85 = distinct !DILexicalBlock(scope: !84, file: !3, line: 3, column: 5)
!86 = !DILocation(line: 72, column: 3, scope: !85)
