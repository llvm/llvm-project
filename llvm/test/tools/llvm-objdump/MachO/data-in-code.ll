; RUN: llc -mtriple x86_64-apple-darwin -filetype=obj -O0 %s -o %t.o
; RUN: llvm-objdump -macho -disassemble -no-show-raw-insn %t.o | FileCheck %s

; CHECK: .long {{[0-9]+}}	@ KIND_JUMP_TABLE32
; CHECK: .long {{[0-9]+}}	@ KIND_JUMP_TABLE32
; CHECK: .long {{[0-9]+}}	@ KIND_JUMP_TABLE32
; CHECK: .long {{[0-9]+}}	@ KIND_JUMP_TABLE32
; CHECK-NOT: invalid instruction encoding
; CHECK-NOT: <unknown>

; ModuleID = '-'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noinline nounwind optnone ssp uwtable
define void @switchfunc(i32 %i) {
  switch i32 %i, label %out [
    i32 0, label %case1
    i32 1, label %case2
    i32 2, label %case3
    i32 3, label %case4
  ]

case1:
  call void @foo()
  br label %out

case2:
  call void @bar()
  br label %out

case3:
  call void @foo()
  br label %out

case4:
  call void @bar()
  br label %out

out:
  ret void
}

declare void @foo()
declare void @bar()
