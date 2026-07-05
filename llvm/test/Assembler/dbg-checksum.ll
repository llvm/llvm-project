; Test that DIFile(checksumkind, checksum) can round-trip through bitcode.
;
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

!llvm.dbg.cu = !{!1, !2, !3, !4}
!llvm.module.flags = !{!9}

!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !7, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !8, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
; CHECK: !DIFile(filename: "a.h", directory: "/test")
!5 = !DIFile(filename: "a.h", directory: "/test")
; CHECK: !DIFile(filename: "b.h", directory: "/test", checksumkind: CSK_MD5, checksum: "595f44fec1e92a71d3e9e77456ba80d1")
!6 = !DIFile(filename: "b.h", directory: "/test", checksumkind: CSK_MD5, checksum: "595f44fec1e92a71d3e9e77456ba80d1")
; CHECK: !DIFile(filename: "c.h", directory: "/test", checksumkind: CSK_SHA1, checksum: "d5db29cd03a2ed055086cef9c31c252b4587d6d0")
!7 = !DIFile(filename: "c.h", directory: "/test", checksumkind: CSK_SHA1, checksum: "d5db29cd03a2ed055086cef9c31c252b4587d6d0")
; CHECK: !DIFile(filename: "d.h", directory: "/test", checksumkind: CSK_SHA256, checksum: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
!8 = !DIFile(filename: "d.h", directory: "/test", checksumkind: CSK_SHA256, checksum: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
!9 = !{i32 2, !"Debug Info Version", i32 3}
