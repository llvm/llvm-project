; RUN: llvm-link -S %s -o - | FileCheck %s

; CHECK-DAG: @foo = private externally_initialized global ptr null
@foo = private externally_initialized global ptr null

@useFoo = global ptr @foo

; CHECK-DAG: @array = appending global [7 x i8] c"abcdefg", align 1
@array = appending global [7 x i8] c"abcdefg", align 1

