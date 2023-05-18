; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK-NOT: error:
!0 = distinct !DICompileUnit(language: 65535,
                             file: !DIFile(filename: "a", directory: "b"))

; CHECK: <stdin>:[[@LINE+1]]:40: error: value for 'language' too large, limit is 65535
!1 = distinct !DICompileUnit(language: 65536,
                             file: !DIFile(filename: "a", directory: "b"))
