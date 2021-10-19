; RUN: llvm-link %s -S -o - | FileCheck %s

; Check that llvm-link does not crash when materializing appending global with
; initializer depending on another appending global.

; CHECK-DAG: @use = appending global [1 x i8*] [i8* bitcast ([1 x i8*]* @var to i8*)]
; CHECK-DAG: @var = appending global [1 x i8*] undef

@use = appending global [1 x i8*] [i8* bitcast ([1 x i8*]* @var to i8*)]
@var = appending global [1 x i8*] undef
