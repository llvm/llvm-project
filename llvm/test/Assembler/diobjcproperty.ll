; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !4}
!named = !{!0, !1, !2, !3, !4, !5}

!0 = distinct !{}
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!2 = !DICompositeType(tag: DW_TAG_structure_type, name: "Object")

; CHECK: !2 = !DICompositeType({{.*}})
; CHECK-NEXT: !3 = !DIObjCProperty(name: "foo", file: !1, line: 7, setter: "setFoo", getter: "getFoo", attributes: 7, type: !2)
!3 = !DIObjCProperty(name: "foo", file: !1, line: 7, setter: "setFoo",
                     getter: "getFoo", attributes: 7, type: !2)

; CHECK-NEXT: !4 = !DIObjCProperty()
!4 = !DIObjCProperty(name: "", file: null, line: 0, setter: "", getter: "",
                     attributes: 0, type: null)
!5 = !DIObjCProperty()
