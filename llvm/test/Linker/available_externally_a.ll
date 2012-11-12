; RUN: llvm-link %s %p/available_externally_b.ll -S -o - | FileCheck %s

@foo = available_externally unnamed_addr constant i32 0

; CHECK: @foo = hidden unnamed_addr constant i32 0
