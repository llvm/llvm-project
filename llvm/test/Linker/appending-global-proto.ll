; RUN: llvm-link %s %p/Inputs/appending-global.ll -S -o - | FileCheck %s
; RUN: llvm-link %p/Inputs/appending-global.ll %s -S -o - | FileCheck %s

; Checks that we can link global variable with appending linkage with the
; existing external declaration.

; CHECK-DAG: @var = appending global [1 x ptr] undef
; CHECK-DAG: @use = global [1 x ptr] [ptr @var]

@var = external global ptr
@use = global [1 x ptr] [ptr @var]
