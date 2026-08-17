; UNSUPPORTED:  target={{.*}}-aix{{.*}}
;
; RUN: llc -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s
; RUN: llvm-dwarfdump --verify %t.o | FileCheck %s --check-prefix=VERIFY

; VERIFY: No errors.

; A property whose getter forwards to the data member holding its backing
; storage. This models a Swift property wrapper, where `x` is a synthesized
; getter over the stored member `_x`, so a consumer can read the storage
; directly instead of calling the getter.
;
; "Foo" lists the member before the property, "Bar" lists the property before
; the member. Emission must not depend on that order.

; CHECK: DW_TAG_class_type
; CHECK:   DW_AT_name ("Foo")
;
; CHECK:   0x[[FOO_X:[0-9a-f]+]]: DW_TAG_member
; CHECK:     DW_AT_name ("_x")
;
; CHECK:   DW_TAG_property
; CHECK:     DW_AT_name ("x")
; CHECK:     DW_AT_type {{.*}} "Int"
; CHECK:     DW_AT_decl_line (8)
; CHECK:     DW_TAG_property_getter
; CHECK:       DW_AT_property_forward (0x[[FOO_X]] "_x")

; CHECK: DW_TAG_class_type
; CHECK:   DW_AT_name ("Bar")
;
; CHECK:   0x[[BAR_Y:[0-9a-f]+]]: DW_TAG_member
; CHECK:     DW_AT_name ("_y")
;
; CHECK:   DW_TAG_property
; CHECK:     DW_AT_name ("y")
; CHECK:     DW_TAG_property_getter
; CHECK:       DW_AT_property_forward (0x[[BAR_Y]] "_y")

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !3, producer: "hand written", isOptimized: false, emissionKind: FullDebug, retainedTypes: !4)
!3 = !DIFile(filename: "t.swift", directory: "/tmp")
!4 = !{!5, !10}

!5 = !DICompositeType(tag: DW_TAG_class_type, name: "Foo", scope: !3, file: !3, line: 7, size: 64, elements: !6)
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "_x", scope: !5, file: !3, line: 8, baseType: !8, size: 64)
!8 = !DIBasicType(name: "Int", size: 64, encoding: DW_ATE_signed)
!9 = !DIProperty(name: "x", file: !3, line: 8, type: !8, getterForward: !7)

!10 = !DICompositeType(tag: DW_TAG_class_type, name: "Bar", scope: !3, file: !3, line: 12, size: 64, elements: !11)
!11 = !{!13, !12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "_y", scope: !10, file: !3, line: 13, baseType: !8, size: 64)
!13 = !DIProperty(name: "y", file: !3, line: 13, type: !8, getterForward: !12)
