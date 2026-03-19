; RUN: not opt -S < %s 2>&1 | FileCheck %s

;; Test that extraData with MDTuple is only allowed for specific DWARF tags:
;; DW_TAG_inheritance, DW_TAG_member, and DW_TAG_variable

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!1 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; Keep all metadata nodes alive so verifier can check them
!named = !{!1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16}
!2 = !{i32 0}

; Valid: DW_TAG_inheritance with tuple extraData should be accepted
!3 = !DIDerivedType(tag: DW_TAG_inheritance, baseType: !1, size: 32, extraData: !2)

; Valid: DW_TAG_member with tuple extraData should be accepted
!4 = !DIDerivedType(tag: DW_TAG_member, name: "field", baseType: !1, size: 32, extraData: !2)

; Valid: DW_TAG_variable (static member) with tuple extraData should be accepted
!5 = !DIDerivedType(tag: DW_TAG_variable, name: "var", baseType: !1, extraData: !2, flags: DIFlagStaticMember)

; Invalid: Empty tuple should be rejected
!6 = !{}
; CHECK: extraData must be ConstantAsMetadata, MDString, DIObjCProperty, or MDTuple with single ConstantAsMetadata operand
; CHECK-NEXT: !{{[0-9]+}} = !DIDerivedType(tag: DW_TAG_member
!7 = !DIDerivedType(tag: DW_TAG_member, name: "field2", baseType: !1, extraData: !6)

; Invalid: Tuple with multiple operands should be rejected
!8 = !{i32 0, i32 1}
; CHECK: extraData must be ConstantAsMetadata, MDString, DIObjCProperty, or MDTuple with single ConstantAsMetadata operand
; CHECK-NEXT: !{{[0-9]+}} = !DIDerivedType(tag: DW_TAG_member
!9 = !DIDerivedType(tag: DW_TAG_member, name: "field3", baseType: !1, extraData: !8)

; Invalid: Tuple with non-ConstantAsMetadata operand should be rejected
!10 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!11 = !{!10}
; CHECK: extraData must be ConstantAsMetadata, MDString, DIObjCProperty, or MDTuple with single ConstantAsMetadata operand
; CHECK-NEXT: !{{[0-9]+}} = !DIDerivedType(tag: DW_TAG_member
!12 = !DIDerivedType(tag: DW_TAG_member, name: "field4", baseType: !1, extraData: !11)

; Valid: DW_TAG_template_alias with proper template parameters tuple
; Template aliases are handled specially and accept any MDTuple for template parameters
!13 = !DITemplateTypeParameter(name: "T", type: !1)
!14 = !{!13}
!15 = !DIDerivedType(tag: DW_TAG_template_alias, name: "MyAlias", baseType: !1, extraData: !14)

; Invalid: DW_TAG_template_alias with non-tuple extraData should fail
; CHECK: invalid template parameters
; CHECK-NEXT: !{{[0-9]+}} = !DIDerivedType(tag: DW_TAG_template_alias
!16 = !DIDerivedType(tag: DW_TAG_template_alias, name: "FailingAlias", baseType: !1, extraData: i32 42)

; CHECK: warning: ignoring invalid debug info

