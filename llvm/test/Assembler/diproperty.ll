; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; !5 and !6 are identical, so uniquing collapses them to a single node.
; CHECK: !named = !{!0, !1, !2, !3, !5, !6, !6}
!named = !{!0, !1, !2, !3, !4, !5, !6}

!0 = distinct !{}
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !DICompositeType(tag: DW_TAG_class_type, name: "Foo", elements: !{!4, !5})

; CHECK: !5 = !DIDerivedType(tag: DW_TAG_member, name: "_x", scope: !3, file: !1, line: 8, baseType: !2, size: 32)
!4 = !DIDerivedType(tag: DW_TAG_member, name: "_x", scope: !3, file: !1,
                    line: 8, baseType: !2, size: 32)

; CHECK-NEXT: !6 = !DIProperty(name: "x", file: !1, line: 8, type: !2, backing_storage: !5)
!5 = !DIProperty(name: "x", file: !1, line: 8, type: !2, backing_storage: !4)

!6 = !DIProperty(name: "x", file: !1, line: 8, type: !2, backing_storage: !4)
