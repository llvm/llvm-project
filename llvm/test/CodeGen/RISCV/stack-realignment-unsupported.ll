; RUN: not llc -mtriple=riscv32 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: RISC-V backend can't currently handle functions that need stack realignment and have variable sized objects

declare void @callee(i8*, i32*)

define void @caller(i32 %n) nounwind {
  %1 = alloca i8, i32 %n
  %2 = alloca i32, align 64
  call void @callee(i8* %1, i32 *%2)
  ret void
}
