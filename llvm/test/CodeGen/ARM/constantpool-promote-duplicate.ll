; RUN: llc -mtriple=arm-eabi -relocation-model=static -arm-promote-constant < %s | FileCheck %s

@const1 = private unnamed_addr constant i32 0, align 4
@const2 = private unnamed_addr constant i32 0, align 4

; const1 and const2 both need labels for debug info, but will be coalesced into
; a single constpool entry

; CHECK-LABEL: @test1
; CHECK-DAG: const1:
; CHECK-DAG: const2:
; CHECK: .fnend
define void @test1() {
  %1 = load i32, ptr @const1, align 4
  call void @a(i32 %1)
  %2 = load i32, ptr @const2, align 4
  call void @a(i32 %2)
  ret void
}

declare void @a(i32)
