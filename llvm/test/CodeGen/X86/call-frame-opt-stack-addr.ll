; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx -frame-pointer=all | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx -frame-pointer=all -no-x86-call-frame-opt | FileCheck %s

; Passing the address of a stack local as a stack argument must not be
; rewritten into a push sequence. X86CallFrameOptimization would leave the
; LEA of the frame index inside the call-frame window; PEI then resolves it
; against RSP after alignment padding for the pushes, so the callee receives
; the wrong pointer (PR210756).

declare preserve_allcc void @sink(i64, i64, i64, i64, i64, i64, ptr)

; CHECK-LABEL: test_stack_addr_arg:
; CHECK:        movq    $42, {{.*}}(%rbp)
; Address of the local must stay RBP-relative (or an equivalent correct
; materialization), and must be written into the outgoing arg slot with a
; store — not captured as RSP after call-frame alignment and then pushed.
; CHECK:        leaq    {{.*}}(%rbp), %[[ADDR:.*]]
; CHECK:        movq    %[[ADDR]], {{.*}}(%rsp)
; CHECK-NOT:    pushq
; CHECK:        callq   sink
define preserve_allcc void @test_stack_addr_arg() {
entry:
  %local = alloca [4 x i64], align 8
  store i64 42, ptr %local, align 8
  call preserve_allcc void @sink(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, ptr %local)
  ret void
}
