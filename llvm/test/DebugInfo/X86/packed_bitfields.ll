; RUN: llc -dwarf-version=2 -mtriple x86_64-apple-macosx -O0 -filetype=obj -o %t_2_le.o %s
; RUN: llvm-dwarfdump -v -debug-info %t_2_le.o | FileCheck %s

; Produced at -O0 from:
; struct {
;   char : 3;
;   char a : 6;
; } __attribute__((__packed__)) b;

; Note that DWARF 2 counts bit offsets backwards from the high end of
; the storage unit to the high end of the bit field.

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"a"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_byte_size  {{.*}} (0x01)
; CHECK-NEXT: DW_AT_bit_size   {{.*}} (0x06)
; CHECK-NEXT: DW_AT_bit_offset {{.*}} (0xffffffffffffffff)
; CHECK-NEXT: DW_AT_data_member_location {{.*}} ({{.*}}0x0{{0*}})

; ModuleID = 'repro.c'
source_filename = "repro.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.4.0"

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
