; RUN: opt -S -passes=speculative-execution %s | FileCheck %s

; Check that SpeculativeExecution's considerHoistingFromTo() drops
; the debug location of the hoisted instructions in a certain branch.

define void @ifThen() !dbg !5 {
; CHECK-LABEL: define void @ifThen(
; CHECK-SAME: ) !dbg [[DBG5:![0-9]+]] {
; CHECK-NEXT:    [[X:%.*]] = add i32 2, 3{{$}}
;
  br i1 true, label %a, label %b, !dbg !8

a:                                                ; preds = %0
  %x = add i32 2, 3, !dbg !9
  br label %b, !dbg !10

b:                                                ; preds = %a, %0
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.ll", directory: "/")
!2 = !{i32 4}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "ifThen", linkageName: "ifThen", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
