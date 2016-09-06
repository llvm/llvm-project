; RUN: llc -O0 -filetype=obj -mtriple=armeb-none-freebsd -debugger-tune=lldb %s -o - \
; RUN: | llvm-dwarfdump --debug-dump=info - | FileCheck %s
; Generated from:
;   struct S {
;     int j:5;
;     int k:6;
;     int m:5;
;     int n:8;
;   } s;

target datalayout = "E-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

%struct.S = type { i24 }

@s = common global %struct.S zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 267633)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "bitfield.c", directory: "/Volumes/Data/llvm")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "s", scope: !0, file: !1, line: 6, type: !5, isLocal: false, isDefinition: true, variable: %struct.S* @s)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 32, align: 32, elements: !6)
!6 = !{!7, !9, !10, !11}
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"j"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x00)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !5, file: !1, line: 2, baseType: !8, size: 5, align: 32)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"k"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x05)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "k", scope: !5, file: !1, line: 3, baseType: !8, size: 6, align: 32, offset: 5)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"m"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x0b)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !5, file: !1, line: 4, baseType: !8, size: 5, align: 32, offset: 11)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"n"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_data1]	(0x10)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !5, file: !1, line: 5, baseType: !8, size: 8, align: 32, offset: 16)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"PIC Level", i32 2}
!15 = !{!"clang version 3.9.0 (trunk 267633)"}
