; This checks that .debug_names can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf64 -accel-tables=Dwarf -dwarf-version=5 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; CHECK:     .debug_info contents:
; CHECK-NEXT: 0x00000000:     Compile Unit: {{.+}}, format = DWARF64,
; CHECK:      [[VARDIE:.+]]:  DW_TAG_variable
; CHECK-NEXT:                   DW_AT_name ("foo")
; CHECK:      [[TYPEDIE:.+]]: DW_TAG_base_type
; CHECK-NEXT:                   DW_AT_name ("int")

; CHECK:      .debug_names contents:
; CHECK-NEXT: Name Index @ 0x0 {
; CHECK-NEXT:   Header {
; CHECK:          Format: DWARF64
; CHECK-NEXT:     Version: 5
; CHECK-NEXT:     CU count: 1
; CHECK-NEXT:     Local TU count: 0
; CHECK-NEXT:     Foreign TU count: 0
; CHECK-NEXT:     Bucket count: 2
; CHECK-NEXT:     Name count: 2
; CHECK:        }
; CHECK-NEXT:   Compilation Unit offsets [
; CHECK-NEXT:     CU[0]: 0x00000000
; CHECK-NEXT:   ]
; CHECK-NEXT:   Abbreviations [
; CHECK-NEXT:     Abbreviation 0x34 {
; CHECK-NEXT:       Tag: DW_TAG_variable
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation 0x24 {
; CHECK-NEXT:       Tag: DW_TAG_base_type
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 0 [
; CHECK-NEXT:     Name 1 {
; CHECK-NEXT:       Hash: 0xB888030
; CHECK-NEXT:       String: {{.+}} "int"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: 0x24
; CHECK-NEXT:         Tag: DW_TAG_base_type
; CHECK-NEXT:         DW_IDX_die_offset: [[TYPEDIE]]
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 1 [
; CHECK-NEXT:     Name 2 {
; CHECK-NEXT:       Hash: 0xB887389
; CHECK-NEXT:       String: {{.+}} "foo"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: 0x34
; CHECK-NEXT:         Tag: DW_TAG_variable
; CHECK-NEXT:         DW_IDX_die_offset: [[VARDIE]]
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT: }

; VERIFY: No errors.

; IR generated and reduced from:
; $ cat foo.c
; int foo;
; $ clang -g -gpubnames -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false)
!3 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
