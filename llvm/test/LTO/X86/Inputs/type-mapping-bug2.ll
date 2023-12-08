target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @c() !dbg !6 {
  unreachable
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"ThinLTO", i32 0}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, retainedTypes: !4)
!3 = !DIFile(filename: "f2", directory: "")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_class_type, file: !3, flags: DIFlagFwdDecl, identifier: "SHARED")
!6 = distinct !DISubprogram(unit: !2)
