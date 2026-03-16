; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj -o %t %s
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s

; Test that DW_OP_LLVM_implicit_pointer is correctly lowered to
; DW_OP_implicit_pointer in DWARF 5 output, with an artificial variable
; DIE describing the dereferenced value.

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name ("foo")
; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location (DW_OP_implicit_pointer
; CHECK:        DW_AT_name ("p")
; CHECK:        DW_AT_type {{.*}} "int *"

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_artificial (true)
; CHECK-NEXT:   DW_AT_type {{.*}} "int"
; CHECK-NEXT:   DW_AT_location (DW_OP_reg5 RDI)

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name ("bar")
; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location (DW_OP_implicit_pointer
; CHECK:        DW_AT_name ("q")

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_artificial (true)
; CHECK-NEXT:   DW_AT_type {{.*}} "int"
; CHECK-NEXT:   DW_AT_const_value (42)

define internal i32 @foo(i32 noundef %p.0.val) !dbg !7 {
entry:
    #dbg_value(i32 %p.0.val, !12, !DIExpression(DW_OP_LLVM_implicit_pointer), !14)
  %add = add nsw i32 %p.0.val, 5, !dbg !15
  ret i32 %add, !dbg !16
}

define internal i32 @bar() !dbg !17 {
entry:
    #dbg_value(i32 42, !20, !DIExpression(DW_OP_LLVM_implicit_pointer), !21)
  ret i32 47, !dbg !22
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

!17 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !18, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!18 = !DISubroutineType(types: !9)
!19 = !{!20}
!20 = !DILocalVariable(name: "q", arg: 1, scope: !17, file: !1, line: 5, type: !10)
!21 = !DILocation(line: 5, scope: !17)
!22 = !DILocation(line: 5, column: 10, scope: !17)