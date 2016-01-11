; RUN: not llvm-lto foobar 2>&1 | FileCheck %s
; CHECK: llvm-lto: error loading file 'foobar': No such file or directory
