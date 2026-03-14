; Verify that if we have two split type units, one with source locations and
; one without, the one without locations doesn't have a DW_AT_stmt_list
; attribute, but the other one does and the .debug_line.dwo section is present.

; RUN: llc -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo \
; RUN:     -dwarf-version=5 -generate-type-units \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu < %s
; RUN: llvm-dwarfdump -v %t.dwo | FileCheck %s

; Currently the no-source-location type comes out first.
; CHECK: .debug_info.dwo contents:
; CHECK: 0x00000000: Type Unit: {{.*}} name = 'S'
; CHECK-SAME: (next unit at [[TU2:0x[0-9a-f]+]])
; CHECK: DW_TAG_type_unit
; CHECK-NOT: DW_AT_stmt_list
; CHECK-NOT: DW_AT_decl_file
; CHECK: [[TU2]]: Type Unit: {{.*}} name = 'T'
; CHECK: DW_TAG_type_unit
; CHECK: DW_AT_stmt_list
; CHECK: DW_AT_decl_file

; CHECK: .debug_line.dwo
; CHECK-NOT: standard_opcode_lengths
; CHECK: file_names[ 0]:
; CHECK-NEXT: name: "t.cpp"
; CHECK-NEXT: dir_index: 0

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32 }
%struct.T = type { i32 }

@s = global %struct.S zeroinitializer, align 4, !dbg !0
@t = global %struct.T zeroinitializer, align 4, !dbg !14

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 (trunk 295942)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "/home/probinson/projects/scratch")
!4 = !{}
!5 = !{!0,!14}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", size: 32, elements: !7, identifier: "_ZTS1S")
!7 = !{}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 5.0.0 (trunk 295942)"}
!11 = distinct !DIGlobalVariable(name: "t", scope: !2, file: !3, line: 10, type: !12, isLocal: false, isDefinition: true)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", file: !3, line: 8, size: 32, elements: !13, identifier: "_ZTS1S")
!13 = !{}
!14 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
