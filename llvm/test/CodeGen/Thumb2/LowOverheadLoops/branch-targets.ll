; RUN: llc -mtriple=thumbv8.1m.main -mattr=+lob -disable-arm-loloops=false -start-after=hardware-loops -stop-before=arm-low-overhead-loops %s -o - | FileCheck %s --check-prefix=CHECK-MID

; Test that the branch targets are correct after isel, even though the loop
; will sometimes be reverted anyway.

; CHECK-MID: name: check_loop_dec_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp ne i32 %count.next, 0
  br i1 %cmp, label %for.header, label %for.cond.cleanup

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_ugt_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_ugt_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp ugt i32 %count.next, 0
  br i1 %cmp, label %for.header, label %for.cond.cleanup

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_ult_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_ult_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp ult i32 %count.next, 1
  br i1 %cmp, label %for.cond.cleanup, label %for.header

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_ult_xor_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_ult_xor_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp ult i32 %count.next, 1
  %negate = xor i1 %cmp, 1
  br i1 %negate, label %for.header, label %for.cond.cleanup

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_sgt_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_sgt_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp sgt i32 %count.next, 0
  br i1 %cmp, label %for.header, label %for.cond.cleanup

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_sge_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_sge_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp sge i32 %count.next, 1
  br i1 %cmp, label %for.header, label %for.cond.cleanup

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_sge_xor_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_sge_xor_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp sge i32 %count.next, 1
  %negated = xor i1 %cmp, 1
  br i1 %negated, label %for.cond.cleanup, label %for.header

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_uge_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_uge_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp uge i32 %count.next, 1
  br i1 %cmp, label %for.header, label %for.cond.cleanup

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: name: check_loop_dec_uge_xor_brcond_combine
; CHECK-MID: bb.0.entry:
; CHECK-MID:   renamable $lr = t2DoLoopStart killed renamable $r3
; CHECK-MID: bb.1.for.header:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECK-MID:   tB %bb.2, 14
; CHECK-MID: bb.2.for.cond.cleanup:
define void @check_loop_dec_uge_xor_brcond_combine(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %N)
  br label %for.body.preheader

for.body.preheader:
  %scevgep = getelementptr i32, i32* %a, i32 -1
  %scevgep4 = getelementptr i32, i32* %c, i32 -1
  %scevgep8 = getelementptr i32, i32* %b, i32 -1
  br label %for.header

for.body:
  %scevgep11 = getelementptr i32, i32* %lsr.iv9, i32 1
  %ld1 = load i32, i32* %scevgep11, align 4
  %scevgep7 = getelementptr i32, i32* %lsr.iv5, i32 1
  %ld2 = load i32, i32* %scevgep7, align 4
  %mul = mul nsw i32 %ld2, %ld1
  %scevgep3 = getelementptr i32, i32* %lsr.iv1, i32 1
  store i32 %mul, i32* %scevgep3, align 4
  %scevgep2 = getelementptr i32, i32* %lsr.iv1, i32 1
  %scevgep6 = getelementptr i32, i32* %lsr.iv5, i32 1
  %scevgep10 = getelementptr i32, i32* %lsr.iv9, i32 1
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp uge i32 %count.next, 1
  %negated = xor i1 %cmp, 1
  br i1 %negated, label %for.cond.cleanup, label %for.header

for.header:
  %lsr.iv9 = phi i32* [ %scevgep8, %for.body.preheader ], [ %scevgep10, %for.body ]
  %lsr.iv5 = phi i32* [ %scevgep4, %for.body.preheader ], [ %scevgep6, %for.body ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body.preheader ], [ %scevgep2, %for.body ]
  %count = phi i32 [ %start, %for.body.preheader ], [ %count.next, %for.body ]
  br label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-MID: check_negated_xor_wls
; CHECK-MID:   $lr = t2WhileLoopStartLR killed renamable $r2
; CHECK-MID:   tB %bb.1
; CHECK-MID: bb.1.while.body:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECk-MID:   tB %bb.2
; CHECK-MID: bb.2.while.end:
define void @check_negated_xor_wls(i16* nocapture %a, i16* nocapture readonly %b, i32 %N) {
entry:
  %wls = call {i32, i1} @llvm.test.start.loop.iterations.i32(i32 %N)
  %wls0 = extractvalue {i32, i1} %wls, 0
  %wls1 = extractvalue {i32, i1} %wls, 1
  %xor = xor i1 %wls1, 1
  br i1 %xor, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %a.addr.06 = phi i16* [ %incdec.ptr1, %while.body ], [ %a, %while.body.preheader ]
  %b.addr.05 = phi i16* [ %incdec.ptr, %while.body ], [ %b, %while.body.preheader ]
  %count = phi i32 [ %wls0, %while.body.preheader ], [ %count.next, %while.body ]
  %incdec.ptr = getelementptr inbounds i16, i16* %b.addr.05, i32 1
  %ld.b = load i16, i16* %b.addr.05, align 2
  %incdec.ptr1 = getelementptr inbounds i16, i16* %a.addr.06, i32 1
  store i16 %ld.b, i16* %a.addr.06, align 2
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp ne i32 %count.next, 0
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-MID: check_negated_cmp_wls
; CHECK-MID:   $lr = t2WhileLoopStartLR killed renamable $r2
; CHECK-MID:   tB %bb.1
; CHECK-MID: bb.1.while.body:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECk-MID:   tB %bb.2
; CHECK-MID: bb.2.while.end:
define void @check_negated_cmp_wls(i16* nocapture %a, i16* nocapture readonly %b, i32 %N) {
entry:
  %wls = call {i32, i1} @llvm.test.start.loop.iterations.i32(i32 %N)
  %wls0 = extractvalue {i32, i1} %wls, 0
  %wls1 = extractvalue {i32, i1} %wls, 1
  %cmp = icmp ne i1 %wls1, 1
  br i1 %cmp, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %a.addr.06 = phi i16* [ %incdec.ptr1, %while.body ], [ %a, %while.body.preheader ]
  %b.addr.05 = phi i16* [ %incdec.ptr, %while.body ], [ %b, %while.body.preheader ]
  %count = phi i32 [ %wls0, %while.body.preheader ], [ %count.next, %while.body ]
  %incdec.ptr = getelementptr inbounds i16, i16* %b.addr.05, i32 1
  %ld.b = load i16, i16* %b.addr.05, align 2
  %incdec.ptr1 = getelementptr inbounds i16, i16* %a.addr.06, i32 1
  store i16 %ld.b, i16* %a.addr.06, align 2
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp.1 = icmp ne i32 %count.next, 0
  br i1 %cmp.1, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-MID: check_negated_reordered_wls
; CHECK-MID:   $lr = t2WhileLoopStartLR killed renamable $r2
; CHECK-MID:   tB %bb.1
; CHECK-MID: bb.1.while.body:
; CHECK-MID:   renamable $lr = t2LoopEndDec killed renamable $lr, %bb.1
; CHECk-MID:   tB %bb.2
; CHECK-MID: bb.2.while.end:
define void @check_negated_reordered_wls(i16* nocapture %a, i16* nocapture readonly %b, i32 %N) {
entry:
  br label %while

while.body.preheader:
  br label %while.body

while.body:
  %a.addr.06 = phi i16* [ %incdec.ptr1, %while.body ], [ %a, %while.body.preheader ]
  %b.addr.05 = phi i16* [ %incdec.ptr, %while.body ], [ %b, %while.body.preheader ]
  %count = phi i32 [ %wls0, %while.body.preheader ], [ %count.next, %while.body ]
  %incdec.ptr = getelementptr inbounds i16, i16* %b.addr.05, i32 1
  %ld.b = load i16, i16* %b.addr.05, align 2
  %incdec.ptr1 = getelementptr inbounds i16, i16* %a.addr.06, i32 1
  store i16 %ld.b, i16* %a.addr.06, align 2
  %count.next = call i32 @llvm.loop.decrement.reg.i32(i32 %count, i32 1)
  %cmp = icmp ne i32 %count.next, 0
  br i1 %cmp, label %while.body, label %while.end

while:
  %wls = call {i32, i1} @llvm.test.start.loop.iterations.i32(i32 %N)
  %wls0 = extractvalue {i32, i1} %wls, 0
  %wls1 = extractvalue {i32, i1} %wls, 1
  %xor = xor i1 %wls1, 1
  br i1 %xor, label %while.end, label %while.body.preheader

while.end:
  ret void
}

declare i32 @llvm.start.loop.iterations.i32(i32)
declare {i32, i1} @llvm.test.start.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32(i32, i32)
