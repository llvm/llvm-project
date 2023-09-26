; RUN: llc < %s -mtriple=riscv64 | FileCheck %s
; RUN: llc < %s -mtriple=riscv64 -align-loops=16 | FileCheck %s -check-prefix=ALIGN_16
; RUN: llc < %s -mtriple=riscv64 -align-loops=32 | FileCheck %s -check-prefix=ALIGN_32

declare void @foo()

define void @test(i32 %n, i32 %m) nounwind {
; CHECK-LABEL:    test:
; CHECK-NOT:        .p2align
; CHECK:            ret

; ALIGN_16-LABEL: test:
; ALIGN_16:         .p2align 4{{$}}
; ALIGN_16-NEXT:  .LBB0_1: # %outer
; ALIGN_16:         .p2align 4{{$}}
; ALIGN_16-NEXT:  .LBB0_2: # %inner

; ALIGN_32-LABEL: test:
; ALIGN_32:         .p2align 5{{$}}
; ALIGN_32-NEXT:  .LBB0_1: # %outer
; ALIGN_32:         .p2align 5{{$}}
; ALIGN_32-NEXT:  .LBB0_2: # %inner
entry:
  br label %outer

outer:
  %outer.iv = phi i32 [0, %entry], [%outer.iv.next, %outer_bb]
  br label %inner

inner:
  %inner.iv = phi i32 [0, %outer], [%inner.iv.next, %inner]
  call void @foo()
  %inner.iv.next = add i32 %inner.iv, 1
  %inner.cond = icmp ne i32 %inner.iv.next, %m
  br i1 %inner.cond, label %inner, label %outer_bb

outer_bb:
  %outer.iv.next = add i32 %outer.iv, 1
  %outer.cond = icmp ne i32 %outer.iv.next, %n
  br i1 %outer.cond, label %outer, label %exit

exit:
  ret void
}
