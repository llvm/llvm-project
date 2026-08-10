; RUN: true
; Companion for debug-info-version-a.ll.

!llvm.module.flags = !{ !0 }
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 42}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang", isOptimized: false, file: !"I AM UNEXPECTED!")
!2 = !{!"b.c", !""}
!3 = !{}
