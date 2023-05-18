; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK:      invalid template parameter
; CHECK-NEXT: !2 = !DICompositeType(
; CHECK-SAME:                       templateParams: !1
; CHECK-NEXT: !1 = !{!0}
; CHECK-NEXT: !0 = !DIBasicType(

!named = !{!0, !1, !2}
!0 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!1 = !{!0}
!2 = !DICompositeType(tag: DW_TAG_structure_type, name: "IntTy", size: 32, align: 32, templateParams: !1)
