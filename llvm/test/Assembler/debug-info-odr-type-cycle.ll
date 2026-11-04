; RUN: llvm-as %s -o -  | llvm-dis | llvm-as  | llvm-dis | FileCheck %s

; CHECK: !namednode = !{!0}
; CHECK-EMPTY:
; CHECK-NEXT: !0 = !DISubprogram(name: "f", linkageName: "f", scope: !1, type: !3, spFlags: 0)
; CHECK-NEXT: !1 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", elements: !2)
; CHECK-NEXT: !2 = !{!0}
; CHECK-NEXT: !3 = !DISubroutineType(types: !4)
; CHECK-NEXT: !4 = !{null}

!namednode = !{!0}
!0 = !DISubprogram(name: "f", linkageName: "f", scope: !1, type: !2, spFlags: 0)
!1 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", elements: !3)
!2 = !DISubroutineType(types: !4)
!3 = !{!0}
!4 = !{null}
