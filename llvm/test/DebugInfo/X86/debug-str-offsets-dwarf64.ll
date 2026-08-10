; This checks that .debug_str_offsets can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf-version=5 -dwarf64 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info -debug-str -debug-str-offsets -v %t | \
; RUN:   FileCheck %s

; CHECK:      .debug_info contents:
; CHECK-NEXT: Compile Unit: {{.*}}, format = DWARF64,
; CHECK:      DW_TAG_compile_unit [1] *
; CHECK:        DW_AT_producer [DW_FORM_strx1] (indexed (00000000) string = "clang version 12.0.0")
; CHECK:        DW_AT_name [DW_FORM_strx1] (indexed (00000001) string = "foo.c")
; CHECK:        DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x0000000000000010)
; CHECK:        DW_AT_comp_dir [DW_FORM_strx1] (indexed (00000002) string = "/tmp")
; CHECK:      DW_TAG_variable [2]  
; CHECK:        DW_AT_name [DW_FORM_strx1] (indexed (00000003) string = "foo")
; CHECK:      DW_TAG_base_type [3]  
; CHECK:        DW_AT_name [DW_FORM_strx1] (indexed (00000004) string = "int")

; CHECK:      .debug_str contents:
; CHECK-NEXT: 0x00000000: "clang version 12.0.0"
; CHECK-NEXT: 0x00000015: "foo.c"
; CHECK-NEXT: 0x0000001b: "/tmp"
; CHECK-NEXT: 0x00000020: "foo"
; CHECK-NEXT: 0x00000024: "int"

; CHECK:      .debug_str_offsets contents:
; CHECK-NEXT: 0x00000000: Contribution size = 44, Format = DWARF64, Version = 5
; CHECK-NEXT: 0x00000010: 0000000000000000 "clang version 12.0.0"
; CHECK-NEXT: 0x00000018: 0000000000000015 "foo.c"
; CHECK-NEXT: 0x00000020: 000000000000001b "/tmp"
; CHECK-NEXT: 0x00000028: 0000000000000020 "foo"
; CHECK-NEXT: 0x00000030: 0000000000000024 "int"

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
