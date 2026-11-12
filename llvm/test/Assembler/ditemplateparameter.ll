; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !3, !4, !5, !5, !6}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = distinct !{}
!1 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
; CHECK: !1 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)

; CHECK-NEXT: !2 = !DITemplateTypeParameter(name: "Ty", type: !1)
; CHECK-NEXT: !3 = !DITemplateTypeParameter(type: !1)
!2 = !DITemplateTypeParameter(name: "Ty", type: !1)
!3 = !DITemplateTypeParameter(type: !1)
!4 = !DITemplateTypeParameter(name: "", type: !1)

; CHECK-NEXT: !4 = !DITemplateValueParameter(name: "V", type: !1, value: i32 7)
; CHECK-NEXT: !5 = !DITemplateValueParameter(type: !1, value: i32 7)
; CHECK-NEXT: !6 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "param", type: !1, value: !"template")
!5 = !DITemplateValueParameter(name: "V", type: !1, value: i32 7)
!6 = !DITemplateValueParameter(type: !1, value: i32 7)
!7 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter,
                               name: "", type: !1, value: i32 7)
!8 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "param", type: !1, value: !"template")
