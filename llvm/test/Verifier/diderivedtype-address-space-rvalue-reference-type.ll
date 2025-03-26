; RUN: opt -S < %s 2>&1 | FileCheck %s

!named = !{!0, !1}
!0 = !DIBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
; CHECK: !DIDerivedType(tag: DW_TAG_rvalue_reference_type, {{.*}}, addressSpace: 1)
!1 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !0, size: 32, align: 32, addressSpace: 1)
