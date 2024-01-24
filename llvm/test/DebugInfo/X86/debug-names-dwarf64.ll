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
; CHECK:      [[SPDIE:.+]]:   DW_TAG_subprogram
; CHECK:                        DW_AT_name ("func")
; CHECK:      [[LABELDIE:.+]]: DW_TAG_label
; CHECK-NEXT:                   DW_AT_name ("MyLabel")

; CHECK:      .debug_names contents:
; CHECK-NEXT: Name Index @ 0x0 {
; CHECK-NEXT:   Header {
; CHECK:          Format: DWARF64
; CHECK-NEXT:     Version: 5
; CHECK-NEXT:     CU count: 1
; CHECK-NEXT:     Local TU count: 0
; CHECK-NEXT:     Foreign TU count: 0
; CHECK-NEXT:     Bucket count: 4
; CHECK-NEXT:     Name count: 4
; CHECK:        }
; CHECK-NEXT:   Compilation Unit offsets [
; CHECK-NEXT:     CU[0]: 0x00000000
; CHECK-NEXT:   ]
; CHECK-NEXT:   Abbreviations [
; CHECK-NEXT:     Abbreviation [[ABBREV_LABEL:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_label
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:       DW_IDX_parent: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation [[ABBREV:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_base_type
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:       DW_IDX_parent: DW_FORM_flag_present
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation [[ABBREV1:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_variable
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:       DW_IDX_parent: DW_FORM_flag_present
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation [[ABBREV_SP:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_subprogram
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:       DW_IDX_parent: DW_FORM_flag_present
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 0 [
; CHECK-NEXT:     Name 1 {
; CHECK-NEXT:       Hash: 0xB888030
; CHECK-NEXT:       String: {{.+}} "int"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: [[ABBREV]]
; CHECK-NEXT:         Tag: DW_TAG_base_type
; CHECK-NEXT:         DW_IDX_die_offset: [[TYPEDIE]]
; CHECK-NEXT:         DW_IDX_parent: <parent not indexed>
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 1 [
; CHECK-NEXT:     Name 2 {
; CHECK-NEXT:       Hash: 0xB887389
; CHECK-NEXT:       String: {{.+}} "foo"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: [[ABBREV1]]
; CHECK-NEXT:         Tag: DW_TAG_variable
; CHECK-NEXT:         DW_IDX_die_offset: [[VARDIE]]
; CHECK-NEXT:         DW_IDX_parent: <parent not indexed>
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:     Name 3 {
; CHECK-NEXT:       Hash: 0x7C96FE71
; CHECK-NEXT:       String: {{.+}} "func"
; CHECK-NEXT:       Entry @ [[FUNC_ENTRY:0x.+]] {
; CHECK-NEXT:         Abbrev: [[ABBREV_SP]]
; CHECK-NEXT:         Tag: DW_TAG_subprogram
; CHECK-NEXT:         DW_IDX_die_offset: [[SPDIE]]
; CHECK-NEXT:         DW_IDX_parent: <parent not indexed>
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 2 [
; CHECK-NEXT:     EMPTY
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 3 [
; CHECK-NEXT:     Name 4 {
; CHECK-NEXT:       Hash: 0xEC64E52B
; CHECK-NEXT:       String: {{.+}} "MyLabel"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: [[ABBREV_LABEL]]
; CHECK-NEXT:         Tag: DW_TAG_label
; CHECK-NEXT:         DW_IDX_die_offset: [[LABELDIE]]
; CHECK-NEXT:         DW_IDX_parent: Entry @ [[FUNC_ENTRY]]
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT: }

; VERIFY: No errors.

; IR generated and reduced from:
; $ cat foo.c
; int foo;
; void func() {
;   goto MyLabel;
;
; MyLabel:
;   return 1;
; }
; $ clang -g -gpubnames -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 0, align 4, !dbg !0

define void @func() !dbg !11 {
  call void @llvm.dbg.label(metadata !15), !dbg !14
  ret void, !dbg !14
}

declare void @llvm.dbg.label(metadata)

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
!11 = distinct !DISubprogram(name: "func", linkageName: "func", scope: !3, file: !3, line: 2, type: !12,  unit: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 2, column: 13, scope: !11)
!15 = !DILabel(scope: !11, name: "MyLabel", file: !3, line: 5)
