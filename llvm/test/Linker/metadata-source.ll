; RUN: llvm-link %p/Inputs/metadata-source-a.ll %p/Inputs/metadata-source-b.ll -S | FileCheck %s

; CHECK: !DIFile(filename: "a.c", directory: "/home/slinder1/test/link", source: "int a;\0A")
; CHECK: !DIFile(filename: "b.c", directory: "/home/slinder1/test/link", source: "int b;\0A")
