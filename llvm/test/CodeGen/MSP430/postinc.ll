; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430"

define zeroext i16 @add(i16* nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, i16* %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: add:
; CHECK: add.w @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, i16* %arrayidx                     ; <i16> [#uses=1]
  %add = add i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @sub(i16* nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, i16* %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: sub:
; CHECK: sub.w @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, i16* %arrayidx                     ; <i16> [#uses=1]
  %add = sub i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @or(i16* nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, i16* %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: or:
; CHECK: bis.w @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, i16* %arrayidx                     ; <i16> [#uses=1]
  %add = or i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @xor(i16* nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, i16* %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: xor:
; CHECK: xor.w @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, i16* %arrayidx                     ; <i16> [#uses=1]
  %add = xor i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @and(i16* nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, i16* %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: and:
; CHECK: and.w @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, i16* %arrayidx                     ; <i16> [#uses=1]
  %add = and i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

