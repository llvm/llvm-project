!named = !{!0, !1, !2, !3}

!0 = !DIFile(filename: "abc", directory: "/path/to")
!1 = !DICompositeType(tag: DW_TAG_class_type, name: "T2", identifier: "T", file: !0)
!2 = !DICompositeType(tag: DW_TAG_class_type, name: "FwdTDef", identifier: "FwdT", file: !0)
!3 = !DICompositeType(tag: DW_TAG_class_type, flags: DIFlagFwdDecl, name: "BothFwdT2", identifier: "BothFwdT", file: !0)
