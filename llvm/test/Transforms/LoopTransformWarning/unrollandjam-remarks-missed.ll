; RUN: opt < %s -passes=transform-warning -disable-output -pass-remarks-missed=transform-warning -pass-remarks-analysis=transform-warning 2>&1 | FileCheck %s
; RUN: opt < %s -passes=transform-warning -disable-output -pass-remarks-output=%t.yaml
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s


; CHECK: warning: source.cpp:19:5: loop not unroll-and-jammed: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering

; YAML:     --- !Failure
; YAML-NEXT: Pass:            transform-warning
; YAML-NEXT: Name:            FailedRequestedUnrollAndJamming
; YAML-NEXT: DebugLoc:        { File: source.cpp, Line: 19, Column: 5 }
; YAML-NEXT: Function:        _Z17test_array_boundsPiS_i
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'loop not unroll-and-jammed: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering'
; YAML-NEXT: ...

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @_Z17test_array_boundsPiS_i(ptr nocapture %A, ptr nocapture readonly %B, i32 %Length) !dbg !8 {
entry:
  %cmp9 = icmp sgt i32 %Length, 0, !dbg !32
  br i1 %cmp9, label %for.body.preheader, label %for.end, !dbg !32

for.body.preheader:
  br label %for.body, !dbg !35

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv, !dbg !35
  %0 = load i32, ptr %arrayidx, align 4, !dbg !35, !tbaa !18
  %idxprom1 = sext i32 %0 to i64, !dbg !35
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %idxprom1, !dbg !35
  %1 = load i32, ptr %arrayidx2, align 4, !dbg !35, !tbaa !18
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !35
  store i32 %1, ptr %arrayidx4, align 4, !dbg !35, !tbaa !18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !32
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !32
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !32
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !32, !llvm.loop !50

for.end.loopexit:
  br label %for.end

for.end:
  ret void, !dbg !36
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0", isOptimized: true, runtimeVersion: 6, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "source.cpp", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "test", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "source.cpp", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "test_disabled", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = distinct !DISubprogram(name: "test_array_bounds", line: 16, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 16, file: !1, scope: !5, type: !6, retainedNodes: !2)
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.5.0"}
!12 = !DILocation(line: 3, column: 8, scope: !13)
!13 = distinct !DILexicalBlock(line: 3, column: 3, file: !1, scope: !4)
!16 = !DILocation(line: 4, column: 5, scope: !17)
!17 = distinct !DILexicalBlock(line: 3, column: 36, file: !1, scope: !13)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 5, column: 9, scope: !23)
!23 = distinct !DILexicalBlock(line: 5, column: 9, file: !1, scope: !17)
!24 = !DILocation(line: 8, column: 1, scope: !4)
!25 = !DILocation(line: 12, column: 8, scope: !26)
!26 = distinct !DILexicalBlock(line: 12, column: 3, file: !1, scope: !7)
!30 = !DILocation(line: 13, column: 5, scope: !26)
!31 = !DILocation(line: 14, column: 1, scope: !7)
!32 = !DILocation(line: 18, column: 8, scope: !33)
!33 = distinct !DILexicalBlock(line: 18, column: 3, file: !1, scope: !8)
!35 = !DILocation(line: 19, column: 5, scope: !33)
!36 = !DILocation(line: 20, column: 1, scope: !8)
!37 = distinct !DILexicalBlock(line: 24, column: 3, file: !1, scope: !46)
!38 = !DILocation(line: 27, column: 3, scope: !37)
!39 = !DILocation(line: 31, column: 3, scope: !37)
!40 = !DILocation(line: 28, column: 9, scope: !37)
!41 = !DILocation(line: 29, column: 11, scope: !37)
!42 = !DILocation(line: 29, column: 7, scope: !37)
!43 = !DILocation(line: 27, column: 32, scope: !37)
!44 = !DILocation(line: 27, column: 30, scope: !37)
!45 = !DILocation(line: 27, column: 21, scope: !37)
!46 = distinct !DISubprogram(name: "test_multiple_failures", line: 26, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 26, file: !1, scope: !5, type: !6, retainedNodes: !2)

!50 = !{!50, !{!"llvm.loop.unroll_and_jam.enable"}}
