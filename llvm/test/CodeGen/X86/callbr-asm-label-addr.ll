; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define i32 @test1(i32 %x) {
; CHECK-LABEL: test1:
; CHECK:         .quad .Ltmp0
; CHECK-NEXT:    .quad .LBB0_1
; CHECK:         .LBB0_1: # Block address taken
; CHECK-NEXT:    # %bar
; CHECK-NEXT:    # Label of block must be emitted
; CHECK-NEXT:    callq foo
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT:  # %bb.2: # %baz
entry:
  callbr void asm sideeffect ".quad ${0:l}\0A\09.quad ${1:l}", "i,!i,~{dirflag},~{fpsr},~{flags}"(ptr blockaddress(@test1, %baz))
          to label %asm.fallthrough [label %bar]

asm.fallthrough:
  br label %bar

bar:
  %call = tail call i32 @foo(i32 %x)
  br label %baz

baz:
  %call1 = tail call i32 @mux(i32 %call)
  ret i32 %call1
}

declare i32 @foo(i32)

declare i32 @mux(i32)
