; RUN: not opt -S < %s 2>&1 | FileCheck %s

!named = !{!0}
; CHECK: DISubprogram contains null entry in `elements` field
!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", elements: !1)
!1 = !{null}
