; RUN: llc -dwarf-version=5 -debugger-tune=lldb -mtriple x86_64-unknown-linux-gnu -O0 -filetype=obj -o %t_2_le.o %s
; RUN: llvm-dwarfdump -v -debug-info %t_2_le.o | FileCheck %s

; Produced at -O0 from:
; struct __attribute__((packed)) bitfield {
; 	int i:1;
; 	int j:32;
; } bitfield;

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"i"
; CHECK-NOT: DW_TAG_member
; CHECK: DW_AT_bit_size   {{.*}} (0x01)
; CHECK-NEXT: DW_AT_data_bit_offset {{.*}} (0x00)

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"j"
; CHECK-NOT: DW_TAG_member
; CHECK: DW_AT_bit_size   {{.*}} (0x20)
; CHECK-NEXT: DW_AT_data_bit_offset {{.*}} (0x01)

; ModuleID = 'packed_bitfields2.c'
source_filename = "packed_bitfields2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.bitfield = type { [5 x i8] }

@bitfield = dso_local global %struct.bitfield zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "bitfield", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.1", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None, sysroot: "/")
!3 = !DIFile(filename: "packed_bitfields2.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bitfield", file: !3, line: 1, size: 40, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !6, file: !3, line: 2, baseType: !9, size: 1, flags: DIFlagBitField, extraData: i64 0)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !6, file: !3, line: 3, baseType: !9, size: 32, offset: 1, flags: DIFlagBitField, extraData: i64 0)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 7, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{!"clang version 11.0.1"}
