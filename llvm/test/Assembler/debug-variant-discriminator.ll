; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2}
!named = !{!0, !1, !2}

; CHECK: !0 = !DICompositeType(tag: DW_TAG_structure_type, name: "Outer", size: 64, align: 64, identifier: "Outer")
; CHECK-NEXT: !1 = !DICompositeType(tag: DW_TAG_variant_part, scope: !0, size: 64, discriminator: !2)
; CHECK-NEXT: !2 = !DIDerivedType(tag: DW_TAG_member, scope: !1, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial)
; CHECK-NEXT: !3 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!0 = !DICompositeType(tag: DW_TAG_structure_type, name: "Outer", size: 64, align: 64, identifier: "Outer")
!1 = !DICompositeType(tag: DW_TAG_variant_part, scope: !0, size: 64, discriminator: !2)
!2 = !DIDerivedType(tag: DW_TAG_member, scope: !1, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial)
!3 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
