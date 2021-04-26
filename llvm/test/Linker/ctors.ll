; RUN: llvm-link %s %p/Inputs/ctors.ll -S -o - | \
; RUN:   FileCheck --check-prefix=ALL --check-prefix=CHECK1 %s
; RUN: llvm-link %p/Inputs/ctors.ll %s -S -o - | \
; RUN:   FileCheck --check-prefix=ALL --check-prefix=CHECK2 %s

; Test the bitcode writer too. It used to crash.
; RUN: llvm-link %s %p/Inputs/ctors.ll -o %t.bc

; ALL: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @f, i8* @v }]
@v = weak global i8 0
; CHECK1: @v = weak global i8 0
; CHECK2: @v = weak global i8 1

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @f, i8* @v }]

define weak void @f() {
  ret void
}
