; REQUIRES: object-emission
; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj -split-dwarf-file=%t.dwo \
; RUN:   -split-dwarf-output=%t.dwo %s -o %t.o
; RUN: llvm-dwarfdump --debug-info %t.dwo | FileCheck %s

; With split DWARF, the DWO id is computed by hashing the CU's DIE tree
; (DIEHash). The DW_OP_implicit_pointer location block contains a reference to
; another DIE, which DIEHash::hashBlockData used to assume was an integer,
; hitting an assertion. Check that we no longer crash and that the reference in
; the .dwo points at the DW_TAG_dwarf_procedure.

; CHECK:      DW_TAG_formal_parameter
; CHECK-NEXT:   DW_AT_location (DW_OP_{{(GNU_)?}}implicit_pointer 0x[[PROC:[0-9a-f]+]] +0)
; CHECK-NEXT:   DW_AT_name ("p")
; CHECK:      0x{{0*}}[[PROC]]: DW_TAG_dwarf_procedure

define dso_local i32 @foo(i32 %p.0.val) !dbg !10 {
entry:
    #dbg_value(i32 %p.0.val, !13, !DIExpression(DW_OP_LLVM_implicit_pointer), !14)
  %add = add nsw i32 %p.0.val, 5, !dbg !14
  ret i32 %add, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!11 = !DISubroutineType(types: !15)
!12 = !{!13}
!13 = !DILocalVariable(name: "p", arg: 1, scope: !10, file: !1, line: 1, type: !5)
!14 = !DILocation(line: 1, scope: !10)
!15 = !{!4, !5}
