; RUN: opt -S -loop-rotate < %s | FileCheck %s
; RUN: opt -S -loop-rotate -enable-mssa-loop-dependency=true -verify-memoryssa < %s | FileCheck %s

;CHECK-LABEL: func
;CHECK-LABEL: entry
;CHECK-NEXT: tail call void @llvm.dbg.value(metadata i32 %a
;CHECK-NEXT: tail call void @llvm.dbg.value(metadata i32 1, metadata ![[I_VAR:[0-9]+]], metadata !DIExpression())
;CHECK-LABEL: for.body:
;CHECK-NEXT: [[I:%.*]] = phi i32 [ 1, %entry ], [ %inc, %for.body ]
;CHECK-NEXT: tail call void @llvm.dbg.value(metadata i32 [[I]], metadata ![[I_VAR]], metadata !DIExpression())

; CHECK: ![[I_VAR]] = !DILocalVariable(name: "i",{{.*}})

; Function Attrs: noinline nounwind
define void @func(i32 %a) local_unnamed_addr #0 !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !10, metadata !11), !dbg !12
  tail call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !11), !dbg !15
  br label %for.cond, !dbg !16

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  tail call void @llvm.dbg.value(metadata i32 %i.0, metadata !13, metadata !11), !dbg !15
  %cmp = icmp slt i32 %i.0, 10, !dbg !17
  br i1 %cmp, label %for.body, label %for.end, !dbg !20

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %i.0, %a, !dbg !22
  %call = tail call i32 @func2(i32 %i.0, i32 %add) #3, !dbg !24
  %inc = add nsw i32 %i.0, 1, !dbg !25
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !13, metadata !11), !dbg !15
  br label %for.cond, !dbg !27, !llvm.loop !28

for.end:                                          ; preds = %for.cond
  ret void, !dbg !31
}

declare i32 @func2(i32, i32) local_unnamed_addr

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (http://llvm.org/git/clang.git 0f3ed908c1f13f83da4b240f7595eb8d05e0a754) (http://llvm.org/git/llvm.git 8e270f5a6b8ceb0f3ac3ef1ffb83c5e29b44ae68)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "debug-phi.c", directory: "/work/projects/src/tests/debug")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 5.0.0 (http://llvm.org/git/clang.git 0f3ed908c1f13f83da4b240f7595eb8d05e0a754) (http://llvm.org/git/llvm.git 8e270f5a6b8ceb0f3ac3ef1ffb83c5e29b44ae68)"}
!6 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "a", arg: 1, scope: !6, file: !1, line: 2, type: !9)
!11 = !DIExpression()
!12 = !DILocation(line: 2, column: 15, scope: !6)
!13 = !DILocalVariable(name: "i", scope: !14, file: !1, line: 3, type: !9)
!14 = distinct !DILexicalBlock(scope: !6, file: !1, line: 3, column: 3)
!15 = !DILocation(line: 3, column: 11, scope: !14)
!16 = !DILocation(line: 3, column: 7, scope: !14)
!17 = !DILocation(line: 3, column: 20, scope: !18)
!18 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 1)
!19 = distinct !DILexicalBlock(scope: !14, file: !1, line: 3, column: 3)
!20 = !DILocation(line: 3, column: 3, scope: !21)
!21 = !DILexicalBlockFile(scope: !14, file: !1, discriminator: 1)
!22 = !DILocation(line: 4, column: 15, scope: !23)
!23 = distinct !DILexicalBlock(scope: !19, file: !1, line: 3, column: 31)
!24 = !DILocation(line: 4, column: 5, scope: !23)
!25 = !DILocation(line: 3, column: 27, scope: !26)
!26 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 2)
!27 = !DILocation(line: 3, column: 3, scope: !26)
!28 = distinct !{!28, !29, !30}
!29 = !DILocation(line: 3, column: 3, scope: !14)
!30 = !DILocation(line: 5, column: 3, scope: !14)
!31 = !DILocation(line: 6, column: 1, scope: !6)
