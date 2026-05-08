; RUN: llc -mtriple=x86_64-linux-gnu -stop-after=finalize-isel -o - %s | FileCheck %s

; Test that DW_OP_LLVM_implicit_pointer survives ISel.

; CHECK: DBG_VALUE $edi, $noreg, ![[VAR:[0-9]+]], !DIExpression(DW_OP_LLVM_implicit_pointer)

define internal i32 @foo(i32 noundef %p.0.val) !dbg !7 {
entry:
    #dbg_value(i32 %p.0.val, !12, !DIExpression(DW_OP_LLVM_implicit_pointer), !14)
  %add = add nsw i32 %p.0.val, 5, !dbg !15
  ret i32 %add, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!6, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!11 = !{!12}
!12 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocation(line: 2, scope: !7)
!15 = !DILocation(line: 2, column: 20, scope: !7)
!16 = !DILocation(line: 2, column: 10, scope: !7)