; Test that DIFile(checksumkind, checksum) representation in Bitcode does
; not change.
;
; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

!llvm.dbg.cu = !{!1, !2, !3}
!llvm.module.flags = !{!7}

!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !4, producer: "clang version 5.0.1 (tags/RELEASE_501/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, producer: "clang version 5.0.1 (tags/RELEASE_501/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "clang version 5.0.1 (tags/RELEASE_501/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
; CHECK: !DIFile(filename: "a.c", directory: "/test")
!4 = !DIFile(filename: "a.c", directory: "/test", checksumkind: CSK_None, checksum: "")
; CHECK: !DIFile(filename: "b.h", directory: "/test", checksumkind: CSK_MD5, checksum: "595f44fec1e92a71d3e9e77456ba80d1")
!5 = !DIFile(filename: "b.h", directory: "/test", checksumkind: CSK_MD5, checksum: "595f44fec1e92a71d3e9e77456ba80d1")
; CHECK: !DIFile(filename: "c.h", directory: "/test", checksumkind: CSK_SHA1, checksum: "d5db29cd03a2ed055086cef9c31c252b4587d6d0")
!6 = !DIFile(filename: "c.h", directory: "/test", checksumkind: CSK_SHA1, checksum: "d5db29cd03a2ed055086cef9c31c252b4587d6d0")
!7 = !{i32 2, !"Debug Info Version", i32 3}
