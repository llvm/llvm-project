target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"


define void @foo() {
  ret void
}

!named = !{!0, !1, !2, !3}

!0 = !DIFile(filename: "abc", directory: "/path/to")
!1 = !DICompositeType(tag: DW_TAG_class_type, name: "T2", identifier: "T", file: !0)
!2 = !DICompositeType(tag: DW_TAG_class_type, name: "FwdTDef", identifier: "FwdT", file: !0)
!3 = !DICompositeType(tag: DW_TAG_class_type, flags: DIFlagFwdDecl, name: "BothFwdT2", identifier: "BothFwdT", file: !0)
