; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: opt < %s -S | FileCheck %s
; RUN: opt < %s -disable-debug-info-type-map | opt -S | FileCheck %s
; RUN: verify-uselistorder %s

;; This test also runs opt to check opt applies the same odr type debug uniquing,
;; and runs it with -disable-debug-info-type-map to check that the subsequent
;; bitcode parsing run agrees with the output.

; Anchor the order of the nodes.
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17}

; Some basic building blocks.
; CHECK:      !0 = !DIBasicType
; CHECK-NEXT: !1 = !DIFile
; CHECK-NEXT: !2 = !DIFile
!0 = !DIBasicType(tag: DW_TAG_base_type, name: "name", size: 1, align: 2, encoding: DW_ATE_unsigned_char)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = !DIFile(filename: "path/to/other", directory: "/path/to/dir")

; Define an identified type with fields and functions. DISubprograms come
; before DICompositeType scope intentionally to check forward refs work
; with the debug ODR type uniquing infra.
; CHECK-NEXT: !3 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !4, file: !2, type: !5, spFlags: 0)
; CHECK-NEXT: !4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "has-uuid",{{.*}}, identifier: "uuid")
; CHECK-NEXT: !5 = !DISubroutineType(types: !6)
; CHECK-NEXT: !6 = !{null}
; CHECK-NEXT: !7 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !4, file: !1, type: !5, spFlags: 0)
; CHECK-NEXT: !8 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !4, file: !1
; CHECK-NEXT: !9 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !4, file: !1
!3 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !5, file: !1, isDefinition: false, type: !18)
!4 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !5, file: !1, isDefinition: false, type: !18)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "has-uuid", file: !1, line: 2, size: 64, align: 32, identifier: "uuid")
!6 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !5, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !5, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!18 = !DISubroutineType(types: !19)
!19 = !{null}

; Define an un-identified type with fields and functions.
; CHECK-NEXT: !10 = !DICompositeType(tag: DW_TAG_structure_type, name: "no-uuid", file: !1
; CHECK-NEXT: !11 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !10, file: !1
; CHECK-NEXT: !12 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !10, file: !1
; CHECK-NEXT: !13 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !10, file: !1, type: !5, spFlags: 0)
; CHECK-NEXT: !14 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !10, file: !1, type: !5, spFlags: 0)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "no-uuid", file: !1, line: 2, size: 64, align: 32)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !8, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "field2", scope: !8, file: !1, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!11 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !8, file: !1, isDefinition: false, type: !18)
!12 = !DISubprogram(name: "foo", linkageName: "foo2", scope: !8, file: !1, isDefinition: false, type: !18)

; Add duplicate fields and members of "no-uuid" in a different file.  These
; should stick around, since "no-uuid" does not have an "identifier:" field.
; CHECK-NEXT: !15 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !10, file: !2,
; CHECK-NEXT: !16 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !10, file: !2, type: !5, spFlags: 0)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !8, file: !2, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!14 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !8, file: !2, isDefinition: false, type: !18)

; Add duplicate fields and members of "has-uuid" in a different file.  These
; should be merged.
!15 = !DIDerivedType(tag: DW_TAG_member, name: "field1", scope: !5, file: !2, line: 4, baseType: !0, size: 32, align: 32, offset: 32)
!16 = !DISubprogram(name: "foo", linkageName: "foo1", scope: !5, file: !2, isDefinition: false, type: !18)

; CHECK-NEXT: !17 = !{!8, !3}
; CHECK-NOT: !DIDerivedType
; CHECK-NOT: !DISubprogram
!17 = !{!15, !16}
