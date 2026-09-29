; This checks that .debug_info can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf-version=3 -dwarf64 -filetype=obj %s -o %t3
; RUN: llvm-dwarfdump -debug-abbrev -debug-info -v %t3 | \
; RUN:   FileCheck %s --check-prefixes=CHECK,DWARFv3

; RUN: llc -mtriple=x86_64 -dwarf-version=4 -dwarf64 -filetype=obj %s -o %t4
; RUN: llvm-dwarfdump -debug-abbrev -debug-info -v %t4 | \
; RUN:   FileCheck %s --check-prefixes=CHECK,DWARFv4

; CHECK:        .debug_abbrev contents:
; CHECK:        [1] DW_TAG_compile_unit   DW_CHILDREN_yes
; CHECK-NEXT:     DW_AT_producer  DW_FORM_strp
; CHECK-NEXT:     DW_AT_language  DW_FORM_data2
; CHECK-NEXT:     DW_AT_name      DW_FORM_strp
; DWARFv3-NEXT:   DW_AT_stmt_list DW_FORM_data8
; DWARFv4-NEXT:   DW_AT_stmt_list DW_FORM_sec_offset
; CHECK-NEXT:     DW_AT_comp_dir  DW_FORM_strp
; CHECK:        [2] DW_TAG_variable   DW_CHILDREN_no
; CHECK-NEXT:     DW_AT_name      DW_FORM_strp
; CHECK-NEXT:     DW_AT_type      DW_FORM_ref4
; CHECK:        [3] DW_TAG_base_type  DW_CHILDREN_no
; CHECK-NEXT:     DW_AT_name      DW_FORM_strp

; CHECK:        .debug_info contents:
; CHECK:        Compile Unit: length = 0x{{([[:xdigit:]]{16})}}, format = DWARF64,
; CHECK:        DW_TAG_compile_unit [1] *
; CHECK-NEXT:     DW_AT_producer [DW_FORM_strp]   ( .debug_str[0x{{([[:xdigit:]]{16})}}] = "clang version 12.0.0")
; CHECK-NEXT:     DW_AT_language [DW_FORM_data2]  (DW_LANG_C99)
; CHECK-NEXT:     DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{([[:xdigit:]]{16})}}] = "foo.c")
; DWARFv3-NEXT:   DW_AT_stmt_list [DW_FORM_data8] (0x0000000000000000)
; DWARFv4-NEXT:   DW_AT_stmt_list [DW_FORM_sec_offset] (0x0000000000000000)
; CHECK-NEXT:     DW_AT_comp_dir [DW_FORM_strp]   ( .debug_str[0x{{([[:xdigit:]]{16})}}] = "/tmp")
; CHECK:        DW_TAG_variable [2]
; CHECK-NEXT:     DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{([[:xdigit:]]{16})}}] = "foo")
; CHECK-NEXT:     DW_AT_type [DW_FORM_ref4]       (cu + {{.+}} => {{.+}} "int")
; CHECK:        DW_TAG_base_type [3]
; CHECK-NEXT:     DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{([[:xdigit:]]{16})}}] = "int")

; IR generated and reduced from:
; $ cat foo.c
; int foo;
; $ clang -g -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
