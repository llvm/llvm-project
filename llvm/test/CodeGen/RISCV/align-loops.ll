; RUN: llc < %s -mtriple=riscv64 | FileCheck %s

declare void @foo()

; Reference case: without llvm.loop.align metadata, RISCV emits no loop
; alignment.
define void @test_noalign(i32 %n, i32 %m) nounwind {
; CHECK-LABEL:    test_noalign:
; CHECK-NOT:        .p2align
; CHECK:            ret
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

; Each loop is aligned independently via its own llvm.loop.align metadata:
; the outer loop to 16 and the inner loop to 32.
define void @test_peralign(i32 %n, i32 %m) nounwind {
; CHECK-LABEL:    test_peralign:
; CHECK:            .p2align 4{{$}}
; CHECK-NEXT:     .LBB1_1: # %outer
; CHECK:            .p2align 5{{$}}
; CHECK-NEXT:     .LBB1_2: # %inner
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
  br i1 %inner.cond, label %inner, label %outer_bb, !llvm.loop !2

outer_bb:
  %outer.iv.next = add i32 %outer.iv, 1
  %outer.cond = icmp ne i32 %outer.iv.next, %n
  br i1 %outer.cond, label %outer, label %exit, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i32 16}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.align", i32 32}
