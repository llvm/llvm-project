; RUN: llc %s -filetype=obj -mtriple arm64e-apple-darwin -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s

; CHECK:   DW_TAG_structure_type
; CHECK: DW_AT_specification	({{.*}} "BaseType")
; CHECK: DW_AT_name	("SpecificationType")
; CHECK: DW_AT_byte_size	(0x08)

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name	("BaseType")
; CHECK: DW_AT_byte_size	(0x08)

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@p = common global i8* null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 1, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "/tmp/p.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}

!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "BaseType", file: !3, size: 64, identifier: "BaseType")

!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "SpecificationType", file: !3, size: 64, identifier: "SpecificationType", specification: !10)
