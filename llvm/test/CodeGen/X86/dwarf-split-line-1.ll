; Verify that split type units with no source locations don't have a
; DW_AT_stmt_list attribute, and the .debug_line.dwo section is suppressed.

; RUN: llc -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo \
; RUN:     -dwarf-version=5 -generate-type-units \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu < %s
; RUN: llvm-dwarfdump -v %t.dwo | FileCheck %s

; FIXME: V5 wants type units in .debug_info.dwo not .debug_types.dwo.
; CHECK-NOT: .debug_line.dwo
; CHECK: .debug_types.dwo contents:
; CHECK: 0x00000000: Type Unit: {{.*}} version = 0x0005 unit_type = DW_UT_split_type abbr_offset
; CHECK: 0x00000018: DW_TAG_type_unit
; CHECK-NOT: DW_AT_stmt_list
; CHECK-NOT: DW_AT_decl_file
; CHECK-NOT: .debug_line.dwo

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32 }

@s = global %struct.S zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 (trunk 295942)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "/home/probinson/projects/scratch")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", size: 32, elements: !7, identifier: "_ZTS1S")
!7 = !{}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 5.0.0 (trunk 295942)"}
