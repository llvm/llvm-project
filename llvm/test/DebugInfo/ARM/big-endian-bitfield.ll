; RUN: llc -O0 -filetype=obj -mtriple=armeb-none-freebsd -debugger-tune=lldb %s -o - \
; RUN: | llvm-dwarfdump --debug-dump=info - | FileCheck %s
; Generated from:
;   struct S {
;     int j:5;
;     int k:6;
;     int m:5;
;     int n:8;
;   } s;

source_filename = "test/DebugInfo/ARM/big-endian-bitfield.ll"
target datalayout = "E-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

%struct.S = type { i24 }

@s = common global %struct.S zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 6, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 267633)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "bitfield.c", directory: "/Volumes/Data/llvm")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, elements: !7)
!7 = !{!8, !10, !11, !12}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !6, file: !3, line: 2, baseType: !9, size: 5)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"j"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x00)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "k", scope: !6, file: !3, line: 3, baseType: !9, size: 6, offset: 5)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !6, file: !3, line: 4, baseType: !9, size: 5, offset: 11)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !6, file: !3, line: 5, baseType: !9, size: 8, offset: 16)
!13 = !{i32 2, !"Dwarf Version", i32 4}
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"k"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x05)
!14 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"m"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x0b)
!15 = !{i32 1, !"PIC Level", i32 2}
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"n"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x10)
!16 = !{!"clang version 3.9.0 (trunk 267633)"}

