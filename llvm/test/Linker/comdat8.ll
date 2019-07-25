; RUN: not llvm-link %s %p/Inputs/comdat8.ll -S -o - 2>&1 | FileCheck %s

$c1 = comdat largest

@some_name = unnamed_addr constant i32 42, comdat($c1)
@c1 = alias i8, inttoptr (i32 1 to i8*)

; CHECK: COMDAT key involves incomputable alias size.
