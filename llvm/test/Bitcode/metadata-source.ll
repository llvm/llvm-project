; Test that DIFile representation upgrades with introduction of
; optional source field.
;
; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!3}

!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 5.0.1 (tags/RELEASE_501/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
; CHECK-NOT: !DIFile({{.*}}source:{{.*}})
!2 = !DIFile(filename: "a.c", directory: "/test")
!3 = !{i32 2, !"Debug Info Version", i32 3}
