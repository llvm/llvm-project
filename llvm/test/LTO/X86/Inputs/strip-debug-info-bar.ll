target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12"

define void @bar() !dbg !3 {
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2)
!2 = !DIFile(filename: "broken", directory: "")
!3 = distinct !DISubprogram(line: 1000, isDefinition: true)

