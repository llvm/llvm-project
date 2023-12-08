; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !1, !2, !2, !2, !2, !3, !4, !2}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

; CHECK: !0 = !{}
!0 = !{}

; CHECK-NEXT: !1 = !GenericDINode(tag: DW_TAG_entry_point, header: "some\00header", operands: {!0, !2, !2})
!1 = !GenericDINode(tag: 3, header: "some\00header", operands: {!0, !3, !4})
!2 = !GenericDINode(tag: 3, header: "some\00header", operands: {!{}, !3, !4})

; CHECK-NEXT: !2 = !GenericDINode(tag: DW_TAG_entry_point)
!3 = !GenericDINode(tag: 3)
!4 = !GenericDINode(tag: 3, header: "")
!5 = !GenericDINode(tag: 3, operands: {})
!6 = !GenericDINode(tag: 3, header: "", operands: {})

; CHECK-NEXT: !3 = distinct !GenericDINode(tag: DW_TAG_entry_point)
!7 = distinct !GenericDINode(tag: 3)

; CHECK-NEXT: !4 = !GenericDINode(tag: 65535)
!8 = !GenericDINode(tag: 65535)

; CHECK-NOT: !
!9 = !GenericDINode(tag: DW_TAG_entry_point)
