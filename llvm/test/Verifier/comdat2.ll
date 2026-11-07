; RUN: llvm-as %s -disable-output
; RUN: opt -mtriple=x86_64-unknown-linux -disable-output
; RUN: not opt -mtriple=x86_64-pc-win32 %s -disable-output 2>&1 | FileCheck %s

$v = comdat any
@v = private global i32 0, comdat($v)
; CHECK: comdat global value has private linkage
