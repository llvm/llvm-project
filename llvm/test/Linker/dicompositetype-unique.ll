; RUN: llvm-link -S -o - %s %S/Inputs/dicompositetype-unique.ll \
; RUN:   | FileCheck %s -check-prefix CHECK -check-prefix FORWARD
; RUN: llvm-link -S -o - %S/Inputs/dicompositetype-unique.ll %s \
; RUN:   | FileCheck %s -check-prefix CHECK -check-prefix REVERSE
; RUN: llvm-link -disable-debug-info-type-map -S -o - %s %S/Inputs/dicompositetype-unique.ll \
; RUN:   | FileCheck %s -check-prefix NOMAP

; Check that the bitcode reader handles this too.
; RUN: llvm-as -o %t1.bc <%s
; RUN: llvm-as -o %t2.bc <%S/Inputs/dicompositetype-unique.ll
; RUN: llvm-link -S -o - %t1.bc %t2.bc \
; RUN:   | FileCheck %s -check-prefix CHECK -check-prefix FORWARD
; RUN: llvm-link -S -o - %t2.bc %t1.bc \
; RUN:   | FileCheck %s -check-prefix CHECK -check-prefix REVERSE
; RUN: llvm-link -disable-debug-info-type-map -S -o - %t1.bc %t2.bc \
; RUN:   | FileCheck %s -check-prefix NOMAP

; Check that the type map will unique two DICompositeTypes.

; CHECK:   !named = !{!0, !1, !2, !3, !0, !1, !2, !3}
; NOMAP:   !named = !{!0, !1, !2, !3, !0, !4, !5, !6}
!named = !{!0, !1, !2, !3}

; Check both directions.
; CHECK:        !1 = distinct !DICompositeType(
; FORWARD-SAME:                                name: "T1"
; REVERSE-SAME:                                name: "T2"
; CHECK-SAME:                                  identifier: "T"
; CHECK-NOT:       identifier: "T"
; CHECK:        !2 = distinct !DICompositeType(
; CHECK-SAME:                                  name: "FwdTDef"
; CHECK-SAME:                                  identifier: "FwdT"
; CHECK-NOT:       identifier: "FwdT"
; CHECK:        !3 = distinct !DICompositeType(
; FORWARD-SAME:                                name: "BothFwdT1"
; REVERSE-SAME:                                name: "BothFwdT2"
; CHECK-SAME:                                  identifier: "BothFwdT"
; CHECK-NOT:       identifier: "BothFwdT"

; These types are different, so we should get both copies when there is no map.
; NOMAP:        !1 = !DICompositeType(
; NOMAP-SAME:                         name: "T1"
; NOMAP-SAME:                         identifier: "T"
; NOMAP:        !2 = !DICompositeType(
; NOMAP-SAME:                         name: "FwdTFwd"
; NOMAP-SAME:                         identifier: "FwdT"
; NOMAP:        !3 = !DICompositeType(
; NOMAP-SAME:                         name: "BothFwdT1"
; NOMAP-SAME:                         identifier: "BothFwdT"
; NOMAP:        !4 = !DICompositeType(
; NOMAP-SAME:                         name: "T2"
; NOMAP-SAME:                         identifier: "T"
; NOMAP-NOT:       identifier: "T"
; NOMAP:        !5 = !DICompositeType(
; NOMAP-SAME:                         name: "FwdTDef"
; NOMAP-SAME:                         identifier: "FwdT"
; NOMAP-NOT:       identifier: "FwdT"
; NOMAP:        !6 = !DICompositeType(
; NOMAP-SAME:                         name: "BothFwdT2"
; NOMAP-SAME:                         identifier: "BothFwdT"
; NOMAP-NOT:       identifier: "BothFwdT"
!0 = !DIFile(filename: "abc", directory: "/path/to")
!1 = !DICompositeType(tag: DW_TAG_class_type, name: "T1", identifier: "T", file: !0)
!2 = !DICompositeType(tag: DW_TAG_class_type, flags: DIFlagFwdDecl, name: "FwdTFwd", identifier: "FwdT", file: !0)
!3 = !DICompositeType(tag: DW_TAG_class_type, flags: DIFlagFwdDecl, name: "BothFwdT1", identifier: "BothFwdT", file: !0)
