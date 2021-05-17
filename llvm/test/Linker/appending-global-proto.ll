; RUN: llvm-link %s %p/Inputs/appending-global.ll -S -o - | FileCheck %s
; RUN: llvm-link %p/Inputs/appending-global.ll %s -S -o - | FileCheck %s

; Checks that we can link global variable with appending linkage with the
; existing external declaration.

; CHECK-DAG: @var = appending global [1 x i8*] undef
; CHECK-DAG: @use = global [1 x i8*] [i8* bitcast ([1 x i8*]* @var to i8*)]

@var = external global i8*
@use = global [1 x i8*] [i8* bitcast (i8** @var to i8*)]
