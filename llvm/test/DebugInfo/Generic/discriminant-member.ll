; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Check for a variant part that has two members, where one uses a list
; of discriminants, and where both refer to a DIE that is not a child
; of the variant.

; CHECK: DW_AT_name [DW_FORM_str{{[a-z]+}}]  ({{(.* = )?}}"Discr")
; CHECK: DW_TAG_variant_part
;   CHECK-NOT: TAG
;     CHECK: DW_AT_discr [DW_FORM_ref4] (cu + {{0x[0-9a-fA-F]+}} => {[[OFFSET:0x[0-9a-fA-F]+]]})
;     CHECK: DW_TAG_variant
;       CHECK: DW_AT_discr_list [DW_FORM_block1] (<0x05> 00 17 01 61 6c )
;       CHECK: DW_TAG_member
;         CHECK: DW_AT_name [DW_FORM_str{{[a-z]+}}]  ({{(.* = )?}}"var0")
;         CHECK: DW_AT_type
;         CHECK: DW_AT_alignment
;         CHECK: DW_AT_data_member_location [DW_FORM_data1]	(0x00)
;     CHECK: DW_TAG_variant
;       CHECK: DW_AT_discr_value [DW_FORM_data1] (0x4b)
;       CHECK: DW_TAG_member
;         CHECK: DW_AT_type
;         CHECK: DW_AT_alignment
;         CHECK: DW_AT_data_member_location [DW_FORM_data1]	(0x00)

%F = type { [0 x i8], ptr, [8 x i8] }
%"F::Nope" = type {}

define internal void @_ZN2e34main17h934ff72f9a38d4bbE() unnamed_addr #0 !dbg !5 {
start:
  %qq = alloca %F, align 8
  call void @llvm.dbg.declare(metadata ptr %qq, metadata !10, metadata !28), !dbg !29
  store ptr null, ptr %qq, !dbg !29
  ret void, !dbg !30
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
!10 = !DILocalVariable(name: "qq", scope: !11, file: !3, line: 3, type: !12, align: 8)
!11 = distinct !DILexicalBlock(scope: !5, file: !3, line: 3, column: 4)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "F", scope: !6, file: !7, size: 128, align: 64, elements: !13, identifier: "7ce1efff6b82281ab9ceb730566e7e20")
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "Discr", scope: !12, file: !7, baseType: !27, size: 64, align: 64)
!15 = !DICompositeType(tag: DW_TAG_variant_part, scope: !12, file: !7, size: 128, align: 64, elements: !16, identifier: "7ce1efff6b82281ab9ceb730566e7e20", discriminator: !14)
!16 = !{!17, !24}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "var0", scope: !15, file: !7, baseType: !18, size: 128, align: 64, extraData: [4 x i32] [i32 23, i32 23, i32 97, i32 108])
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "Yep", scope: !12, file: !7, size: 128, align: 64, elements: !19, identifier: "7ce1efff6b82281ab9ceb730566e7e20::Yep")
!19 = !{!20, !22}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "var1", scope: !18, file: !7, baseType: !21, size: 8, align: 8, offset: 64)
!21 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "var2", scope: !18, file: !7, baseType: !23, size: 64, align: 64)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&u8", baseType: !21, size: 64, align: 64)
!24 = !DIDerivedType(tag: DW_TAG_member, scope: !15, file: !7, baseType: !25, size: 128, align: 64, extraData: i32 75)
!25 = !DICompositeType(tag: DW_TAG_structure_type, name: "Nope", scope: !12, file: !7, size: 128, align: 64, elements: !4, identifier: "7ce1efff6b82281ab9ceb730566e7e20::Nope")
!27 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!28 = !DIExpression()
!29 = !DILocation(line: 3, scope: !11)
!30 = !DILocation(line: 4, scope: !5)
