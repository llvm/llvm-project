; RUN: opt -S -structurizecfg %s -o - | FileCheck %s

; The structurize cfg pass used to do a post-order traversal to generate a list
; of ; basic blocks and then operate on the list in reverse.  This led to bugs,
; because sometimes successors would be visited before their predecessors.
; The fix for this was to do a reverse post-order traversal which is what the
; algorithm requires.

; Function Attrs: nounwind
define void @test(float* nocapture %out, i32 %K1, float* nocapture readonly %nr) {

; CHECK: entry:
; CHECK: br label %for.body
entry:
  br label %for.body

; CHECK: for.body:
; CHECK: br i1 %{{[0-9]+}}, label %lor.lhs.false, label %Flow
for.body:                                         ; preds = %for.body.backedge, %entry
  %indvars.iv = phi i64 [ %indvars.iv.be, %for.body.backedge ], [ 1, %entry ]
  %best_val.027 = phi float [ %best_val.027.be, %for.body.backedge ], [ 5.000000e+01, %entry ]
  %prev_start.026 = phi i32 [ %tmp26, %for.body.backedge ], [ 0, %entry ]
  %best_count.025 = phi i32 [ %best_count.025.be, %for.body.backedge ], [ 0, %entry ]
  %tmp0 = trunc i64 %indvars.iv to i32
  %cmp1 = icmp eq i32 %tmp0, %K1
  br i1 %cmp1, label %if.then, label %lor.lhs.false

; CHECK: lor.lhs.false:
; CHECK: br label %Flow
lor.lhs.false:                                    ; preds = %for.body
  %arrayidx = getelementptr inbounds float, float* %nr, i64 %indvars.iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = add nsw i64 %indvars.iv, -1
  %arrayidx2 = getelementptr inbounds float, float* %nr, i64 %tmp2
  %tmp3 = load float, float* %arrayidx2, align 4
  %cmp3 = fcmp une float %tmp1, %tmp3
  br i1 %cmp3, label %if.then, label %for.body.1

; CHECK: Flow:
; CHECK: br i1 %{{[0-9]+}}, label %if.then, label %Flow1

; CHECK: if.then:
; CHECK: br label %Flow1
if.then:                                          ; preds = %lor.lhs.false, %for.body
  %sub4 = sub nsw i32 %tmp0, %prev_start.026
  %tmp4 = add nsw i64 %indvars.iv, -1
  %arrayidx8 = getelementptr inbounds float, float* %nr, i64 %tmp4
  %tmp5 = load float, float* %arrayidx8, align 4
  br i1 %cmp1, label %for.end, label %for.body.1

; CHECK: for.end:
; CHECK: ret void
for.end:                                          ; preds = %for.body.1, %if.then
  %best_val.0.lcssa = phi float [ %best_val.233, %for.body.1 ], [ %tmp5, %if.then ]
  store float %best_val.0.lcssa, float* %out, align 4
  ret void

; CHECK: Flow1
; CHECK: br i1 %{{[0-9]}}, label %for.body.1, label %Flow2

; CHECK: for.body.1:
; CHECK: br i1 %{{[0-9]+}}, label %for.body.6, label %Flow3
for.body.1:                                       ; preds = %if.then, %lor.lhs.false
  %best_val.233 = phi float [ %tmp5, %if.then ], [ %best_val.027, %lor.lhs.false ]
  %best_count.231 = phi i32 [ %sub4, %if.then ], [ %best_count.025, %lor.lhs.false ]
  %indvars.iv.next.454 = add nsw i64 %indvars.iv, 5
  %tmp22 = trunc i64 %indvars.iv.next.454 to i32
  %cmp1.5 = icmp eq i32 %tmp22, %K1
  br i1 %cmp1.5, label %for.end, label %for.body.6

; CHECK: Flow2:
; CHECK: br i1 %{{[0-9]+}}, label %for.end, label %for.body

; CHECK: for.body.6:
; CHECK: br i1 %cmp5.6, label %if.then6.6, label %for.body.backedge
for.body.6:                                       ; preds = %for.body.1
  %indvars.iv.next.559 = add nsw i64 %indvars.iv, 6
  %tmp26 = trunc i64 %indvars.iv.next.559 to i32
  %sub4.6 = sub nsw i32 %tmp26, %tmp22
  %cmp5.6 = icmp slt i32 %best_count.231, %sub4.6
  br i1 %cmp5.6, label %if.then6.6, label %for.body.backedge

; CHECK: if.then6.6
; CHECK: br label %for.body.backedge
if.then6.6:                                       ; preds = %for.body.6
  %arrayidx8.6 = getelementptr inbounds float, float* %nr, i64 %indvars.iv.next.454
  %tmp29 = load float, float* %arrayidx8.6, align 4
  br label %for.body.backedge

; CHECK: Flow3:
; CHECK: br label %Flow2

; CHECK: for.body.backedge:
; CHECK: br label %Flow3
for.body.backedge:                                ; preds = %if.then6.6, %for.body.6
  %best_val.027.be = phi float [ %tmp29, %if.then6.6 ], [ %best_val.233, %for.body.6 ]
  %best_count.025.be = phi i32 [ %sub4.6, %if.then6.6 ], [ %best_count.231, %for.body.6 ]
  %indvars.iv.be = add nsw i64 %indvars.iv, 7
  br label %for.body
}
