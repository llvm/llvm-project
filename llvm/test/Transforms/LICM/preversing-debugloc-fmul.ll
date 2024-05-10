; RUN: opt -passes=licm -verify-memoryssa -S < %s | FileCheck %s

; Test that LICM's hoistRegin() 

define zeroext i1 @invariant_denom(double %v) !dbg !5 {
; CHECK:  loop:
; CHECK:    [[TMP1:%.*]] = fmul fast double {{.*}}, !dbg [[DBG29:![0-9]+]]
; CHECK:  [[DBG29]] = !DILocation(line: 5,
;
entry:
  br label %loop, !dbg !25

loop:                                             ; preds = %loop, %entry
  %v3 = phi i32 [ 0, %entry ], [ %v11, %loop ], !dbg !26
  %v4 = phi i32 [ 0, %entry ], [ %v12, %loop ], !dbg !27
  %v5 = uitofp i32 %v4 to double, !dbg !28
  %v6 = fdiv fast double %v5, %v, !dbg !29
  %v7 = fptoui double %v6 to i64, !dbg !30
  %v8 = and i64 %v7, 1, !dbg !31
  %v9 = xor i64 %v8, 1, !dbg !32
  %v10 = trunc i64 %v9 to i32, !dbg !33
  %v11 = add i32 %v10, %v3, !dbg !34
  %v12 = add nuw i32 %v4, 1, !dbg !35
  %v13 = icmp eq i32 %v12, -1, !dbg !36
  br i1 %v13, label %end, label %loop, !dbg !37

end:                                              ; preds = %loop
  %v15 = phi i32 [ %v11, %loop ], !dbg !38
  %v16 = icmp ne i32 %v15, 0, !dbg !39
  ret i1 %v16, !dbg !40
}


!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "preserving-debugloc.ll", directory: "/")
!2 = !{i32 16}
!3 = !{i32 13}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "invariant_denom", linkageName: "invariant_denom", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !12, !14, !15, !16, !17, !18, !19, !20, !21, !23, !24}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 3, type: !10)
!12 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !13)
!13 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !13)
!15 = !DILocalVariable(name: "5", scope: !5, file: !1, line: 6, type: !13)
!16 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 7, type: !13)
!17 = !DILocalVariable(name: "7", scope: !5, file: !1, line: 8, type: !13)
!18 = !DILocalVariable(name: "8", scope: !5, file: !1, line: 9, type: !10)
!19 = !DILocalVariable(name: "9", scope: !5, file: !1, line: 10, type: !10)
!20 = !DILocalVariable(name: "10", scope: !5, file: !1, line: 11, type: !10)
!21 = !DILocalVariable(name: "11", scope: !5, file: !1, line: 12, type: !22)
!22 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!23 = !DILocalVariable(name: "12", scope: !5, file: !1, line: 14, type: !10)
!24 = !DILocalVariable(name: "13", scope: !5, file: !1, line: 15, type: !22)
!25 = !DILocation(line: 1, column: 1, scope: !5)
!26 = !DILocation(line: 2, column: 1, scope: !5)
!27 = !DILocation(line: 3, column: 1, scope: !5)
!28 = !DILocation(line: 4, column: 1, scope: !5)
!29 = !DILocation(line: 5, column: 1, scope: !5)
!30 = !DILocation(line: 6, column: 1, scope: !5)
!31 = !DILocation(line: 7, column: 1, scope: !5)
!32 = !DILocation(line: 8, column: 1, scope: !5)
!33 = !DILocation(line: 9, column: 1, scope: !5)
!34 = !DILocation(line: 10, column: 1, scope: !5)
!35 = !DILocation(line: 11, column: 1, scope: !5)
!36 = !DILocation(line: 12, column: 1, scope: !5)
!37 = !DILocation(line: 13, column: 1, scope: !5)
!38 = !DILocation(line: 14, column: 1, scope: !5)
!39 = !DILocation(line: 15, column: 1, scope: !5)
!40 = !DILocation(line: 16, column: 1, scope: !5)
