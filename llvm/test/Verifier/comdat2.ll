; RUN: llvm-as %s -o /dev/null
; RUN: opt -mtriple=x86_64-unknown-linux -o /dev/null
; RUN: not opt -mtriple=x86_64-pc-win32 %s -o /dev/null 2>&1 | FileCheck %s

$v = comdat any
@v = private global i32 0, comdat($v)
; CHECK: comdat global value has private linkage
