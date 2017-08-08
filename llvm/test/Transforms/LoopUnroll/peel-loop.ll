; RUN: opt < %s -S -loop-unroll -unroll-force-peel-count=3 -verify-dom-info -simplifycfg -instcombine | FileCheck %s

; Basic loop peeling - check that we can peel-off the first 3 loop iterations
; when explicitly requested.
; CHECK-LABEL: @basic
; CHECK: %[[CMP0:.*]] = icmp sgt i32 %k, 0
; CHECK: br i1 %[[CMP0]], label %[[NEXT0:.*]], label %for.end
; CHECK: [[NEXT0]]:
; CHECK: store i32 0, i32* %p, align 4
; CHECK: %[[CMP1:.*]] = icmp eq i32 %k, 1
; CHECK: br i1 %[[CMP1]], label %for.end, label %[[NEXT1:.*]]
; CHECK: [[NEXT1]]:
; CHECK: %[[INC1:.*]] = getelementptr inbounds i32, i32* %p, i64 1
; CHECK: store i32 1, i32* %[[INC1]], align 4
; CHECK: %[[CMP2:.*]] = icmp sgt i32 %k, 2
; CHECK: br i1 %[[CMP2]], label %[[NEXT2:.*]], label %for.end
; CHECK: [[NEXT2]]:
; CHECK: %[[INC2:.*]] = getelementptr inbounds i32, i32* %p, i64 2
; CHECK: store i32 2, i32* %[[INC2]], align 4
; CHECK: %[[CMP3:.*]] = icmp eq i32 %k, 3
; CHECK: br i1 %[[CMP3]], label %for.end, label %[[LOOP_PH:.*]]
; CHECK: [[LOOP_PH]]:
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: %[[IV:.*]] = phi i32 [ 3, %[[LOOP_PH]] ], [ {{.*}}, %[[LOOP]] ]

define void @basic(i32* %p, i32 %k) #0 {
entry:
  %cmp3 = icmp slt i32 0, %k
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %p.addr.04 = phi i32* [ %p, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %p.addr.04, i32 1
  store i32 %i.05, i32* %p.addr.04, align 4
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %inc, %k
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

; Make sure peeling works correctly when a value defined in a loop is used
; in later code - we need to correctly plumb the phi depending on which
; iteration is actually used.
; CHECK-LABEL: @output
; CHECK: %[[CMP0:.*]] = icmp sgt i32 %k, 0
; CHECK: br i1 %[[CMP0]], label %[[NEXT0:.*]], label %for.end
; CHECK: [[NEXT0]]:
; CHECK: store i32 0, i32* %p, align 4
; CHECK: %[[CMP1:.*]] = icmp eq i32 %k, 1
; CHECK: br i1 %[[CMP1]], label %for.end, label %[[NEXT1:.*]]
; CHECK: [[NEXT1]]:
; CHECK: %[[INC1:.*]] = getelementptr inbounds i32, i32* %p, i64 1
; CHECK: store i32 1, i32* %[[INC1]], align 4
; CHECK: %[[CMP2:.*]] = icmp sgt i32 %k, 2
; CHECK: br i1 %[[CMP2]], label %[[NEXT2:.*]], label %for.end
; CHECK: [[NEXT2]]:
; CHECK: %[[INC2:.*]] = getelementptr inbounds i32, i32* %p, i64 2
; CHECK: store i32 2, i32* %[[INC2]], align 4
; CHECK: %[[CMP3:.*]] = icmp eq i32 %k, 3
; CHECK: br i1 %[[CMP3]], label %for.end, label %[[LOOP_PH:.*]]
; CHECK: [[LOOP_PH]]:
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: %[[IV:.*]] = phi i32 [ 3, %[[LOOP_PH]] ], [ %[[IV:.*]], %[[LOOP]] ]
; CHECK: %ret = phi i32 [ 0, %entry ], [ 1, %[[NEXT0]] ], [ 2, %[[NEXT1]] ], [ 3, %[[NEXT2]] ], [ %[[IV]], %[[LOOP]] ]
; CHECK: ret i32 %ret
define i32 @output(i32* %p, i32 %k) #0 {
entry:
  %cmp3 = icmp slt i32 0, %k
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %p.addr.04 = phi i32* [ %p, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %p.addr.04, i32 1
  store i32 %i.05, i32* %p.addr.04, align 4
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %inc, %k
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %ret = phi i32 [ 0, %entry], [ %inc, %for.cond.for.end_crit_edge ]
  ret i32 %ret
}
