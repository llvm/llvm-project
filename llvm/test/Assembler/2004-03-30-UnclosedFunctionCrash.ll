; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: found end of file when expecting more instructions

define void @foo() {
