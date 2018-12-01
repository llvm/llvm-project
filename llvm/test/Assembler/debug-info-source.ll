; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Ensure we accept debug info where DIFiles within a DICompileUnit either all
; have source, or none have source.

define dso_local void @foo() !dbg !6 {
  ret void
}

define dso_local void @bar() !dbg !7 {
  ret void
}

define dso_local void @baz() !dbg !9 {
  ret void
}

define dso_local void @qux() !dbg !11 {
  ret void
}

!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
; CHECK: !1 = !DIFile(filename: "foo.c", directory: "dir", source: "void foo() { }\0A")
!1 = !DIFile(filename: "foo.c", directory: "dir", source: "void foo() { }\0A")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3)
; CHECK: !3 = !DIFile(filename: "qux.h", directory: "dir")
!3 = !DIFile(filename: "qux.h", directory: "dir")
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", file: !1, unit: !0)
!7 = distinct !DISubprogram(name: "bar", file: !8, unit: !0)
; CHECK: !8 = !DIFile(filename: "bar.h", directory: "dir", source: "void bar() { }\0A")
!8 = !DIFile(filename: "bar.h", directory: "dir", source: "void bar() { }\0A")
!9 = distinct !DISubprogram(name: "baz", file: !10, unit: !2)
; CHECK: !10 = !DIFile(filename: "baz.c", directory: "dir")
!10 = !DIFile(filename: "baz.c", directory: "dir")
!11 = distinct !DISubprogram(name: "qux", file: !3, unit: !2)
