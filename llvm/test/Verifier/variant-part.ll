; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s

!named = !{!0}
!0 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", size: 64, discriminator: !1)
!1 = !DIDerivedType(tag: DW_TAG_member, scope: !0, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial)
!3 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)

; CHECK: discriminator can only appear on variant part
