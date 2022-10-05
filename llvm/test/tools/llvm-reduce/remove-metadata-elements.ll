; RUN: llvm-reduce %s -o %t --delta-passes=metadata --test FileCheck --test-arg %s --test-arg --input-file
; RUN: FileCheck %s < %t --implicit-check-not="boring"

; Test that debug metadata lists can be reduced by making sure debug info for
; "boring" globals are removed.

; $ cat a.c
; int boringA, interesting, boringB;
; $ clang a.c -g -S -emit-llvm -o -

@A = dso_local global i32 0, align 4, !dbg !0
@B = dso_local global i32 0, align 4, !dbg !5
@C = dso_local global i32 0, align 4, !dbg !8

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "boringA", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "a.c", directory: "")
!4 = !{!0, !5, !8}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
; CHECK: !DIGlobalVariable(name: "interesting"
!6 = distinct !DIGlobalVariable(name: "interesting", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "boringB", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 2}
