; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!1}
; CHECK: DICompileUnit(language: DW_LANG_C99, file: !{{[0-9]+}}, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "foo.c", directory: "/path/to/dir")
