; RUN: %llc_dwarf %s -o - -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Generated from:
; namespace normal { inline namespace inlined { int i; } }
; Check that an inline namespace is emitted with DW_AT_export_symbols

; CHECK: DW_TAG_namespace
; CHECK-NEXT:   DW_AT_name {{.*}} "normal"
; CHECK-NOT:    DW_AT_export_symbols
; CHECK-NOT:    NULL
; CHECK:        DW_TAG_namespace
; CHECK-NEXT:     DW_AT_name {{.*}} "inlined"
; CHECK-NOT:      DW_TAG
; CHECK-NOT:      NULL
; CHECK:          DW_AT_export_symbols [DW_FORM_flag_present]	(true)
; CHECK-NOT:      DW_TAG
; CHECK:             DW_TAG_variable
; CHECK-NEXT:        DW_AT_name {{.*}} "i"

source_filename = "namespace.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@_ZN6normal7inlined1iE = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DIGlobalVariable(name: "i", linkageName: "_ZN6normal7inlined1iE", scope: !1, file: !2, line: 1, type: !4, isLocal: false, isDefinition: true)
!1 = !DINamespace(name: "inlined", scope: !3, file: !2, line: 1, exportSymbols: true)
!2 = !DIFile(filename: "namespace.cpp", directory: "/")
!3 = !DINamespace(name: "normal", scope: null, file: !2, line: 1)
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 4.0.0 (trunk 285825) (llvm/trunk 285822)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, globals: !7)
!6 = !{}
!7 = !{!0}
!8 = !{i32 2, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 4.0.0 (trunk 285825) (llvm/trunk 285822)"}
