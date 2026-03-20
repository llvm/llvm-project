; RUN: opt -passes=argpromotion -S < %s | \
; RUN:   llc -mtriple=x86_64-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s

; End-to-end test: ArgumentPromotion preserves debug info via
; DW_OP_implicit_pointer, which the backend lowers to DWARF.

; CHECK: DW_TAG_formal_parameter
; CHECK:   DW_AT_location (DW_OP_implicit_pointer
; CHECK:   DW_AT_name ("p")
; CHECK: DW_TAG_dwarf_procedure

define internal i32 @foo(ptr %p) !dbg !7 {
entry:
    #dbg_value(ptr %p, !12, !DIExpression(), !14)
  %val = load i32, ptr %p, align 4, !dbg !15
  %add = add nsw i32 %val, 5, !dbg !15
  ret i32 %add, !dbg !16
}

define i32 @bar(i32 %a) !dbg !17 {
entry:
  %x = alloca i32, align 4, !dbg !20
  store i32 %a, ptr %x, align 4, !dbg !20
  %result = call i32 @foo(ptr %x), !dbg !20
  ret i32 %result, !dbg !20
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

!17 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !18, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!18 = !DISubroutineType(types: !19)
!19 = !{!6, !6}
!20 = !DILocation(line: 6, scope: !17)