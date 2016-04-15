; RUN: llvm-link %s %p/debug-info-version-b.ll -S -o - | FileCheck %s

; Test linking of incompatible debug info versions. The debug info
; from the other file should be dropped.

; CHECK-NOT: !DIFile(filename: "b.c", directory: "")
; CHECK:     !DIFile(filename: "a.c", directory: "")
; CHECK-NOT: !DIFile(filename: "b.c", directory: "")

!llvm.module.flags = !{ !0 }
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang", isOptimized: true, emissionKind: FullDebug, file: !2, enums: !3, retainedTypes: !3)
!2 = !DIFile(filename: "a.c", directory: "")
!3 = !{}
