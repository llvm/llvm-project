; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}

; Produced at -O0 from:
; struct {
;   char : 3;
;   char a : 6;
; } __attribute__((__packed__)) b;

; Note that DWARF 2 counts bit offsets backwards from the high end of
; the storage unit to the high end of the bit field.

; CHECK: .section .debug_info
; CHECK:      .b8 3    // Abbrev {{.*}} DW_TAG_structure_type
; CHECK:      .b8 3    // DW_AT_decl_line
; CHECK-NEXT: .b8 1    // DW_AT_byte_size
; CHECK-NEXT: .b8 6    // DW_AT_bit_size
; Negative offset must be encoded as an unsigned integer.
; CHECK-NEXT: .b8 127  // DW_AT_bit_offset
; CHECK-NEXT: .b8 2    // DW_AT_data_member_location

%struct.anon = type { i16 }

@b = global %struct.anon zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None, sysroot: "/")
!3 = !DIFile(filename: "repro.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 16, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, line: 3, baseType: !9, size: 6, offset: 3, flags: DIFlagBitField, extraData: i64 0)
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !{i32 7, !"Dwarf Version", i32 2}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = !{!"clang version 11.0.0 "}
