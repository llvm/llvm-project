; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK:      invalid template parameter
; CHECK-NEXT: !{{[0-9]+}} = !DICompositeType(
; CHECK-SAME:                       templateParams: !{{[0-9]+}}
; CHECK-NEXT: !{{[0-9]+}} = !{!{{[0-9]+}}}
; CHECK-NEXT: !{{[0-9]+}} = !DIBasicType(

!named = !{!0, !1, !2}
!0 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!1 = !{!0}
!2 = !DICompositeType(tag: DW_TAG_structure_type, name: "IntTy", size: 32, align: 32, templateParams: !1)
