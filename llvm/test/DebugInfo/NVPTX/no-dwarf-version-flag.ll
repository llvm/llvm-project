; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s --implicit-check-not=DW_AT_data_bit_offset
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}

; The module has no "Dwarf Version" flag. NVPTX emits DWARF v2 in that case, so
; the DWARF v2 bitfield encoding (DW_AT_bit_offset) must be used and not the
; DWARF v4 DW_AT_data_bit_offset.

; CHECK: .b8 2 // DWARF version number
; CHECK: // DW_AT_bit_offset

%struct.S = type { i8 }

@g = global %struct.S zeroinitializer, !dbg !0

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !6, file: !7, line: 1, type: !8, isLocal: false, isDefinition: true)
!6 = distinct !DICompileUnit(language: DW_LANG_C99, file: !7, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !15)
!7 = !DIFile(filename: "bitfield.c", directory: "/test")
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !7, line: 1, size: 8, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !7, line: 2, baseType: !11, size: 3, offset: 1, flags: DIFlagBitField, extraData: i64 0)
!11 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!0}
