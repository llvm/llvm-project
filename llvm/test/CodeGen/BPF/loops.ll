; RUN: llc < %s -mtriple=bpfel -mcpu=v1 | FileCheck %s

define zeroext i16 @add(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <ptr> [#uses=1]
; CHECK-LABEL: add:
; CHECK: r{{[0-9]+}} += r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = add i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @sub(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <ptr> [#uses=1]
; CHECK-LABEL: sub:
; CHECK: r{{[0-9]+}} -= r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = sub i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @or(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <ptr> [#uses=1]
; CHECK-LABEL: or:
; CHECK: r{{[0-9]+}} |= r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = or i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @xor(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <ptr> [#uses=1]
; CHECK-LABEL: xor:
; CHECK: r{{[0-9]+}} ^= r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = xor i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @and(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <ptr> [#uses=1]
; CHECK-LABEL: and:
; CHECK: r{{[0-9]+}} &= r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = and i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}
