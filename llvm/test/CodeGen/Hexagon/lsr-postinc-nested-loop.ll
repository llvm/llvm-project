; RUN: llc -O3 -march=hexagon < %s | FileCheck %s
; Test to ensure LSR does not optimize out addrec of the outerloop.
; This will help to generate post-increment instructions, otherwise
; it end up an as extra reg+reg add inside the loop.
; CHECK:  loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: memuh{{.*}}++
; CHECK: endloop


define dso_local signext i16 @foo(ptr nocapture readonly %filt, ptr nocapture readonly %inp, i32 %c1, i32 %c2) local_unnamed_addr {
entry:
  %cmp28 = icmp sgt i32 %c1, 0
  %cmp221 = icmp sgt i32 %c2, 0
  %or.cond = and i1 %cmp28, %cmp221
  br i1 %or.cond, label %for.cond1.preheader.us, label %for.cond.cleanup

for.cond1.preheader.us:                           ; preds = %entry, %for.cond1.for.cond.cleanup3_crit_edge.us
  %filt.addr.032.us = phi ptr [ %scevgep, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %filt, %entry ]
  %inp.addr.031.us = phi ptr [ %scevgep35, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ %inp, %entry ]
  %l.030.us = phi i32 [ %inc11.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %entry ]
  %sum0.029.us = phi i16 [ %add8.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %entry ]
  %scevgep = getelementptr i16, ptr %filt.addr.032.us, i32 %c2
  br label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.cond1.preheader.us
  %z.025.us = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc.us, %for.body4.us ]
  %filt.addr.124.us = phi ptr [ %filt.addr.032.us, %for.cond1.preheader.us ], [ %incdec.ptr.us, %for.body4.us ]
  %inp.addr.123.us = phi ptr [ %inp.addr.031.us, %for.cond1.preheader.us ], [ %incdec.ptr5.us, %for.body4.us ]
  %sum0.122.us = phi i16 [ %sum0.029.us, %for.cond1.preheader.us ], [ %add8.us, %for.body4.us ]
  %incdec.ptr.us = getelementptr inbounds i16, ptr %filt.addr.124.us, i32 1
  %0 = load i16, ptr %filt.addr.124.us, align 2
  %incdec.ptr5.us = getelementptr inbounds i16, ptr %inp.addr.123.us, i32 1
  %1 = load i16, ptr %inp.addr.123.us, align 2
  %add.us = add i16 %0, %sum0.122.us
  %add8.us = add i16 %add.us, %1
  %inc.us = add nuw nsw i32 %z.025.us, 1
  %exitcond = icmp eq i32 %inc.us, %c2
  br i1 %exitcond, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us:         ; preds = %for.body4.us
  %scevgep35 = getelementptr i16, ptr %inp.addr.031.us, i32 %c2
  %inc11.us = add nuw nsw i32 %l.030.us, 1
  %exitcond36 = icmp eq i32 %inc11.us, %c1
  br i1 %exitcond36, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %entry
  %sum0.0.lcssa = phi i16 [ 0, %entry ], [ %add8.us, %for.cond1.for.cond.cleanup3_crit_edge.us ]
  ret i16 %sum0.0.lcssa
}
