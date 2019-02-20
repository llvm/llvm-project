; RUN: llc -filetype=obj -thread-model=single -mtriple=wasm32-unknown-unknown -o %t.o %s
; RUN: llvm-nm %t.o | FileCheck %s

@foo = internal global i32 1, align 4
@bar = global i32 1, align 4

; CHECK: 00000004 D bar
; CHECK: 00000000 d foo
