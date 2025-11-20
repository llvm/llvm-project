; RUN: %llc_dwarf -debugger-tune=lldb -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: dsymutil %t -o %t.dSYM
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=SAME-NAME %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=DIFFERENT-NAME %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=UNIQUE-DIFFERENT-NAME %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s


; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name ("SameName")
; CHECK: DW_AT_linkage_name ("SameName")

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name ("DifferentName")
; CHECK: DW_AT_linkage_name ("UniqueDifferentName")

; The name count should be 5 (the two variables, "int", "SameName", "DifferentName", "UniqueDifferentName").
; SAME-NAME: Name count: 6

; The accelarator should only have one entry for the three following names.
; SAME-NAME: "SameName" 
; SAME-NAME-NOT: "SameName" 

; DIFFERENT-NAME: "DifferentName"
; DIFFERENT-NAME-NOT: "DifferentName"

; UNIQUE-DIFFERENT-NAME: "UniqueDifferentName"
; UNIQUE-DIFFERENT-NAME-NOT: "UniqueDifferentName"

; Verification should succeed.
; VERIFY: No errors.

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx14.5.0"

%struct.SameName = type { i32 }
%struct.DifferentName = type { i32 }

@q = global %struct.SameName zeroinitializer, align 4, !dbg !0
@r = global %struct.DifferentName zeroinitializer, align 4, !dbg !5

!llvm.module.flags = !{!14, !15, !16, !17, !18, !19, !20}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 9, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 1", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "r", scope: !2, file: !3, line: 10, type: !7, isLocal: false, isDefinition: true)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "DifferentName", file: !3, line: 5, size: 32, runtimeLang: DW_LANG_Swift, identifier: "UniqueDifferentName", elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !7, file: !3, line: 6, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "SameName", file: !3, line: 1, size: 32, runtimeLang: DW_LANG_Swift, identifier: "SameName", elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !11, file: !3, line: 2, baseType: !10, size: 32)
!14 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 5]}
!15 = !{i32 7, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 8, !"PIC Level", i32 2}
!19 = !{i32 7, !"uwtable", i32 1}
!20 = !{i32 7, !"frame-pointer", i32 1}
!21 = !{!"clang version 1"}
