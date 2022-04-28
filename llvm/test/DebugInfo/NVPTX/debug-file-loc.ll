; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

; // Bitcode int this test case is reduced version of compiled code below:
;extern "C" {
;#line 1 "/source/dir/foo.h"
;__device__ void foo() {}
;#line 2 "/source/dir/bar.cu"
;__device__ void bar() {}
;}

; CHECK: .target sm_{{[0-9]+}}, debug

; CHECK: .visible .func foo()
; CHECK: .loc [[FOO:[0-9]+]] 1 31
; CHECK:  ret;
; CHECK: .visible .func bar()
; CHECK: .loc [[BAR:[0-9]+]] 2 31
; CHECK:  ret;

define void @foo() !dbg !4 {
bb:
  ret void, !dbg !10
}

define void @bar() !dbg !7 {
bb:
  ret void, !dbg !11
}

; CHECK-DAG: .file [[FOO]] "/source/dir{{[/\\]}}foo.h"
; CHECK-DAG: .file [[BAR]] "/source/dir{{[/\\]}}bar.cu"
; CHECK: .section .debug_abbrev
; CHECK-NEXT: {
; CHECK-NEXT: .b8 1                                // Abbreviation Code
; CHECK-NEXT: .b8 17                               // DW_TAG_compile_unit
; CHECK-NEXT: .b8 0                                // DW_CHILDREN_no
; CHECK-NEXT: .b8 37                               // DW_AT_producer
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 19                               // DW_AT_language
; CHECK-NEXT: .b8 5                                // DW_FORM_data2
; CHECK-NEXT: .b8 3                                // DW_AT_name
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 16                               // DW_AT_stmt_list
; CHECK-NEXT: .b8 6                                // DW_FORM_data4
; CHECK-NEXT: .b8 27                               // DW_AT_comp_dir
; CHECK-NEXT: .b8 8                                // DW_FORM_string
; CHECK-NEXT: .b8 17                               // DW_AT_low_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 18                               // DW_AT_high_pc
; CHECK-NEXT: .b8 1                                // DW_FORM_addr
; CHECK-NEXT: .b8 0                                // EOM(1)
; CHECK-NEXT: .b8 0                                // EOM(2)
; CHECK-NEXT: .b8 0                                // EOM(3)
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_info
; CHECK-NEXT: {
; CHECK-NEXT: .b32 50                              // Length of Unit
; CHECK-NEXT: .b8 2                                // DWARF version number
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_abbrev                   // Offset Into Abbrev. Section
; CHECK-NEXT: .b8 8                                // Address Size (in bytes)
; CHECK-NEXT: .b8 1                                // Abbrev [1] 0xb:0x2b DW_TAG_compile_unit
; CHECK-NEXT: .b8 0                                // DW_AT_producer
; CHECK-NEXT: .b8 4                                // DW_AT_language
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b8 98                               // DW_AT_name
; CHECK-NEXT: .b8 97
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 46
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b32 .debug_line                     // DW_AT_stmt_list
; CHECK-NEXT: .b8 47                               // DW_AT_comp_dir
; CHECK-NEXT: .b8 115
; CHECK-NEXT: .b8 111
; CHECK-NEXT: .b8 117
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 99
; CHECK-NEXT: .b8 101
; CHECK-NEXT: .b8 47
; CHECK-NEXT: .b8 100
; CHECK-NEXT: .b8 105
; CHECK-NEXT: .b8 114
; CHECK-NEXT: .b8 0
; CHECK-NEXT: .b64 Lfunc_begin0                    // DW_AT_low_pc
; CHECK-NEXT: .b64 Lfunc_end1                      // DW_AT_high_pc
; CHECK-NEXT: }
; CHECK-NEXT: .section .debug_loc { }
; CHECK-NOT: debug_

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "bar.cu", directory: "/source/dir")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DIFile(filename: "foo.h", directory: "/source/dir")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocation(line: 1, column: 31, scope: !4)
!11 = !DILocation(line: 2, column: 31, scope: !7)
