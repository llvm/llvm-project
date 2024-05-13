; RUN: llvm-link %s -S -o - | FileCheck %s

; Check that llvm-link does not crash when materializing appending global with
; initializer depending on another appending global.

; CHECK-DAG: @use = appending global [1 x ptr] [ptr @var]
; CHECK-DAG: @var = appending global [1 x ptr] undef

@use = appending global [1 x ptr] [ptr @var]
@var = appending global [1 x ptr] undef
