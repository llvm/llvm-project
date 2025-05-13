; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Check for a variant part where a variant has multiple members.

; CHECK: DW_AT_name [DW_FORM_str{{[a-z]+}}]  ({{.*}} = "Discr")
; CHECK: DW_TAG_variant_part
;   CHECK-NOT: TAG
;     CHECK: DW_AT_discr [DW_FORM_ref4] (cu + {{0x[0-9a-fA-F]+}} => {[[OFFSET:0x[0-9a-fA-F]+]]})
;     CHECK: DW_TAG_variant
;       CHECK: DW_AT_discr_value [DW_FORM_data1] (0x4a)
;       CHECK: DW_TAG_member
;         CHECK: DW_AT_name [DW_FORM_str{{[a-z]+}}]  ({{.*}} = "field0")
;         CHECK: DW_AT_type
;         CHECK: DW_AT_alignment
;         CHECK: DW_AT_data_member_location [DW_FORM_data1]	(0x00)
;       CHECK: DW_TAG_member
;         CHECK: DW_AT_name [DW_FORM_str{{[a-z]+}}]  ({{.*}} = "field1")
;         CHECK: DW_AT_type
;         CHECK: DW_AT_alignment
;         CHECK: DW_AT_data_member_location [DW_FORM_data1]	(0x08)
;     CHECK: DW_TAG_variant
;       CHECK: DW_AT_discr_value [DW_FORM_data1] (0x4b)
;       CHECK: DW_TAG_member
;         CHECK: DW_AT_name [DW_FORM_str{{[a-z]+}}]  ({{.*}} = "field2")
;         CHECK: DW_AT_type
;         CHECK: DW_AT_alignment
;         CHECK: DW_AT_data_member_location [DW_FORM_data1]	(0x00)

%F = type { [0 x i8], ptr, [8 x i8] }

define internal void @_ZN2e34main17h934ff72f9a38d4bbE() unnamed_addr #0 !dbg !5 {
start:
  %qq = alloca %F, align 8
  call void @llvm.dbg.declare(metadata ptr %qq, metadata !10, metadata !24), !dbg !25
  store ptr null, ptr %qq, !dbg !25
  ret void, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

attributes #0 = { nounwind uwtable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 1, !"PIE Level", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !3, producer: "gnat-llvm", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4)
!3 = !DIFile(filename: "e3.rs", directory: "/home/tromey/Ada")
!4 = !{}
!5 = distinct !DISubprogram(name: "main", linkageName: "_ZN2e34mainE", scope: !6, file: !3, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagMainSubprogram, isOptimized: false, unit: !2, templateParams: !4, retainedNodes: !4)
!6 = !DINamespace(name: "e3", scope: null)
!7 = !DIFile(filename: "<unknown>", directory: "")
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "qq", scope: !11, file: !3, line: 3, type: !12, align: 64)
!11 = distinct !DILexicalBlock(scope: !5, file: !3, line: 3, column: 4)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "F", scope: !6, file: !7, size: 128, align: 64, elements: !13, identifier: "7ce1efff6b82281ab9ceb730566e7e20")
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "Discr", scope: !12, file: !7, baseType: !23, size: 64, align: 64)
!15 = !DICompositeType(tag: DW_TAG_variant_part, scope: !12, file: !7, size: 128, align: 64, elements: !16, identifier: "7ce1efff6b82281ab9ceb730566e7e20", discriminator: !14)
!16 = !{!17, !22}
!17 = !DIDerivedType(tag: DW_TAG_member, scope: !15, file: !7, baseType: !18, size: 128, align: 64, extraData: i32 74)
!18 = !DICompositeType(tag: DW_TAG_variant, scope: !15, file: !7, size: 128, align: 64, elements: !19)
!19 = !{!20, !21}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "field0", scope: !18, file: !7, baseType: !23, size: 64, align: 64, offset: 0)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !18, file: !7, baseType: !23, size: 64, align: 64, offset: 64)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !15, file: !7, baseType: !23, size: 64, align: 64, offset: 0, extraData: i32 75)
!23 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!24 = !DIExpression()
!25 = !DILocation(line: 3, scope: !11)
!26 = !DILocation(line: 4, scope: !5)
