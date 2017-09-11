; RUN: %llc_dwarf -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -v -debug-info %t.o | FileCheck %s
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
; CHECK:          DW_AT_bit_offset {{.*}} (0xfffffffffffffff8)
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x0)
source_filename = "test/DebugInfo/ARM/bitfield.ll"
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

%struct.anon = type { i8, [5 x i8] }

@a = common global %struct.anon zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "test.i", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 48, align: 8, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !6, file: !3, line: 2, baseType: !9, size: 8, align: 8)
!9 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", scope: !6, file: !3, line: 4, baseType: !11, size: 28, align: 32, offset: 12)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 1, !"min_enum_size", i32 4}
!16 = !{i32 1, !"PIC Level", i32 2}
!17 = !{!"clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)"}

