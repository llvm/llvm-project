; RUN: opt < %s -passes=unpredictable-profile-loader -unpredictable-hints-file=%S/Inputs/mispredict.prof -unpredictable-hints-frequency-profile=%S/Inputs/frequency.prof -unpredictable-hints-min-ratio=0.1 -S | FileCheck %s
; RUN: opt < %s -passes=unpredictable-profile-loader -unpredictable-hints-file=%S/Inputs/mispredict.prof -unpredictable-hints-frequency-profile=%S/Inputs/frequency.prof -unpredictable-hints-min-ratio=0.5 -S | FileCheck --check-prefixes=MIN %s

; CHECK-LABEL: @sel_arr
; MIN-LABEL:   @sel_arr
define void @sel_arr(ptr %dst, ptr %s1, ptr %s2, ptr %s3) !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata ptr %dst, metadata !14, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata ptr %s1, metadata !15, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata ptr %s2, metadata !16, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata ptr %s3, metadata !17, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.value(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !24
  br label %for.body, !dbg !25

for.cond.cleanup:                                 ; preds = %for.body
  ret void, !dbg !26

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %latch ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !18, metadata !DIExpression()), !dbg !24
  %arrayidx = getelementptr inbounds i32, ptr %s1, i64 %indvars.iv, !dbg !27
  %0 = load i32, ptr %arrayidx, align 4, !dbg !27
  %cmp1 = icmp slt i32 %0, 10035, !dbg !27
; CHECK: br i1 %cmp1, label %if.then, label %if.else
; CHECK-SAME: !unpredictable
; MIN: br i1 %cmp1, label %if.then, label %if.else
; MIN-NOT: !unpredictable
  br i1 %cmp1, label %if.then, label %if.else, !dbg !27

if.then:
  %then.cond = getelementptr inbounds i32, ptr %s2, i64 %indvars.iv, !dbg !27
  call void @llvm.dbg.value(metadata ptr %then.cond, metadata !20, metadata !DIExpression()), !dbg !33
  %1 = load i32, ptr %then.cond, align 4, !dbg !34
  br label %latch

if.else:
  %else.cond = getelementptr inbounds i32, ptr %s3, i64 %indvars.iv, !dbg !27
  call void @llvm.dbg.value(metadata ptr %else.cond, metadata !20, metadata !DIExpression()), !dbg !33
  %2 = load i32, ptr %else.cond, align 4, !dbg !34
  br label %latch

latch:
  %3 = phi i32 [ %1, %if.then ], [ %2, %if.else ]
  %arrayidx8 = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv, !dbg !35
  store i32 %3, ptr %arrayidx8, align 4, !dbg !36
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !37
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !18, metadata !DIExpression()), !dbg !24
  %exitcond.not = icmp eq i64 %indvars.iv.next, 20000, !dbg !38
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !dbg !25
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1)
!1 = !DIFile(filename: "3.c", directory: "/test")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "sel_arr", scope: !1, file: !1, line: 28, type: !9, scopeLine: 28, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !11, !11, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15, !16, !17, !18, !20}
!14 = !DILocalVariable(name: "dst", arg: 1, scope: !8, file: !1, line: 28, type: !11)
!15 = !DILocalVariable(name: "s1", arg: 2, scope: !8, file: !1, line: 28, type: !11)
!16 = !DILocalVariable(name: "s2", arg: 3, scope: !8, file: !1, line: 28, type: !11)
!17 = !DILocalVariable(name: "s3", arg: 4, scope: !8, file: !1, line: 28, type: !11)
!18 = !DILocalVariable(name: "i", scope: !19, file: !1, line: 38, type: !12)
!19 = distinct !DILexicalBlock(scope: !8, file: !1, line: 38, column: 5)
!20 = !DILocalVariable(name: "p", scope: !21, file: !1, line: 39, type: !11)
!21 = distinct !DILexicalBlock(scope: !22, file: !1, line: 38, column: 33)
!22 = distinct !DILexicalBlock(scope: !19, file: !1, line: 38, column: 5)
!23 = !DILocation(line: 0, scope: !8)
!24 = !DILocation(line: 0, scope: !19)
!25 = !DILocation(line: 38, column: 5, scope: !19)
!26 = !DILocation(line: 42, column: 1, scope: !8)
!27 = !DILocation(line: 39, column: 18, scope: !21)
!33 = !DILocation(line: 0, scope: !21)
!34 = !DILocation(line: 40, column: 18, scope: !21)
!35 = !DILocation(line: 40, column: 9, scope: !21)
!36 = !DILocation(line: 40, column: 16, scope: !21)
!37 = !DILocation(line: 38, column: 29, scope: !22)
!38 = !DILocation(line: 38, column: 23, scope: !22)
