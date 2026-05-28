;; RUN: llc --filetype=obj -o - %s | dxil-dis -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

;; CHECK: define i32 @dbgassign(i32 %a, i32 %b) {
;; CHECK-NEXT: entry:
;; CHECK-NEXT:   call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata [[LV:![0-9]+]], metadata [[EX:![0-9]+]])
;; CHECK-NEXT:   %add = add nsw i32 %a, %b
;; CHECK-NEXT:   call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata [[LV]], metadata [[EX]])
;; CHECK-NEXT:   ret i32 %add
;; CHECK-NEXT: }

define i32 @dbgassign(i32 %a, i32 %b) !dbg !5 {
entry:
    #dbg_assign(i1 poison, !10, !DIExpression(), !11, ptr poison, !DIExpression(), !12)
    #dbg_assign(i32 0, !10, !DIExpression(), !13, ptr poison, !DIExpression(), !12)
    #dbg_assign(i32 %a, !10, !DIExpression(), !14, ptr poison, !DIExpression(), !12)
  %add = add nsw i32 %a, %b, !dbg !15
    #dbg_assign(i32 %add, !10, !DIExpression(), !16, ptr poison, !DIExpression(), !12)
  ret i32 %add
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dbgassign.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!5 = distinct !DISubprogram(name: "dbgassign", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9, keyInstructions: true)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "result", scope: !5, file: !1, line: 3, type: !8)
!11 = distinct !DIAssignID()
!12 = !DILocation(line: 0, scope: !5)
!13 = distinct !DIAssignID()
!14 = distinct !DIAssignID()
!15 = !DILocation(line: 7, column: 10, scope: !5, atomGroup: 3, atomRank: 2)
!16 = distinct !DIAssignID()
