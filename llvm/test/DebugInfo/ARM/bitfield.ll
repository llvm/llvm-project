; RUN: %llc_dwarf -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s
; REQUIRES: object-emission
;
; Generated from:
; struct {
;   char c;
;   int : 4;
;   int reserved : 28;
; } a;
;
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "reserved"
; CHECK:          DW_AT_byte_size  {{.*}} (0x04)
; CHECK:          DW_AT_bit_size   {{.*}} (0x1c)
; CHECK:          DW_AT_bit_offset {{.*}} (0x18)
; CHECK:          DW_AT_data_member_location {{.*}}00
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

%struct.anon = type { i8, [5 x i8] }

@a = common global %struct.anon zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "test.i", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "a", scope: !0, file: !1, line: 5, type: !5, isLocal: false, isDefinition: true, variable: %struct.anon* @a)
!5 = !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 1, size: 48, align: 8, elements: !6)
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !5, file: !1, line: 2, baseType: !8, size: 8, align: 8)
!8 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", scope: !5, file: !1, line: 4, baseType: !10, size: 28, align: 32, offset: 12)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 1, !"min_enum_size", i32 4}
!15 = !{i32 1, !"PIC Level", i32 2}
!16 = !{!"clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)"}
