; RUN: llvm-as < %s 2>&1 >/dev/null | FileCheck %s

; Ensure we reject debug info where the DIFiles of a DICompileUnit mix source
; and no-source.

define dso_local void @foo() !dbg !5 {
  ret void
}

define dso_local void @bar() !dbg !6 {
  ret void
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}

!2 = !DIFile(filename: "foo.c", directory: "dir", source: "void foo() { }\0A")
; CHECK: inconsistent use of embedded source
; CHECK: warning: ignoring invalid debug info
!3 = !DIFile(filename: "bar.h", directory: "dir")

!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2)
!5 = distinct !DISubprogram(name: "foo", file: !2, unit: !4)
!6 = distinct !DISubprogram(name: "bar", file: !3, unit: !4)
