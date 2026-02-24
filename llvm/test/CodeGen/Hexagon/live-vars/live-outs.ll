; RUN: llc -O3 -verify-machineinstrs < %s -o /dev/null
; REQUIRES: asserts
;
; This is a compile-only regression test (asserts build) for Hexagon.

define i32 @foo(ptr nocapture readnone %x, i32 %n, ptr nocapture readonly %p,
                ptr nocapture readonly %q, ptr %b) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %div = lshr i32 %n, 3
  %cmp149 = icmp eq i32 %div, 0
  br i1 %cmp149, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %if.end
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %arrayidx.phi = phi ptr [ %arrayidx.inc, %for.inc ], [ %p, %for.body.preheader ]
  %arrayidx2.phi = phi ptr [ %arrayidx2.inc, %for.inc ], [ %q, %for.body.preheader ]
  %i.050 = phi i32 [ %inc, %for.inc ], [ 0, %for.body.preheader ]
  %0 = load i8, ptr %arrayidx.phi, align 1
  %1 = load i8, ptr %arrayidx2.phi, align 1
  %cmp4 = icmp eq i8 %0, %1
  br i1 %cmp4, label %for.inc, label %for.end.loopexit

for.inc:                                          ; preds = %for.body
  %arrayidx2.inc = getelementptr i8, ptr %arrayidx2.phi, i32 1
  %arrayidx.inc = getelementptr i8, ptr %arrayidx.phi, i32 1
  %inc = add nuw nsw i32 %i.050, 1
  %cmp1 = icmp ult i32 %inc, %div
  br i1 %cmp1, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body, %for.inc
  %i.0.lcssa.ph = phi i32 [ %i.050, %for.body ], [ %inc, %for.inc ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %if.end
  %i.0.lcssa = phi i32 [ 0, %if.end ], [ %i.0.lcssa.ph, %for.end.loopexit ]
  %cmp8 = icmp eq i32 %i.0.lcssa, %div
  br i1 %cmp8, label %if.end30, label %if.then10

if.then10:                                        ; preds = %for.end
  %rem = and i32 %n, 7
  %cmp11 = icmp eq i32 %rem, 0
  br i1 %cmp11, label %return, label %if.end14

if.end14:                                         ; preds = %if.then10
  %sub = sub nsw i32 8, %rem
  %shl = shl i32 1, %sub
  %sub16 = add i32 %shl, 255
  %arrayidx18 = getelementptr inbounds i8, ptr %p, i32 %i.0.lcssa
  %2 = load i8, ptr %arrayidx18, align 1
  %sub16.not = or i32 %sub16, -256
  %neg = xor i32 %sub16.not, 255
  %arrayidx21 = getelementptr inbounds i8, ptr %q, i32 %i.0.lcssa
  %3 = load i8, ptr %arrayidx21, align 1
  %4 = xor i8 %3, %2
  %5 = zext i8 %4 to i32
  %6 = and i32 %5, %neg
  %cmp26 = icmp eq i32 %6, 0
  br i1 %cmp26, label %return, label %if.end30

if.end30:                                         ; preds = %for.end, %if.end14
  %cmp31 = icmp eq ptr %b, null
  br i1 %cmp31, label %return, label %if.then33

if.then33:                                        ; preds = %if.end30
  store i8 0, ptr %b, align 1
  br label %return

return:                                           ; preds = %if.end30, %if.then33, %if.end14, %if.then10, %entry
  %retval.0 = phi i32 [ 0, %entry ], [ 1, %if.then10 ], [ 1, %if.end14 ], [ 0, %if.then33 ], [ 0, %if.end30 ]
  ret i32 %retval.0
}
