; RUN: opt < %s -passes=simple-loop-unswitch -disable-output

; PR38283
; PR38737
define void @f1() {
for.cond1thread-pre-split.lr.ph.lr.ph:
  %tobool4 = icmp eq i16 undef, 0
  br label %for.cond1thread-pre-split

for.cond1thread-pre-split:                        ; preds = %if.end, %for.cond1thread-pre-split.lr.ph.lr.ph
  %tobool3 = icmp eq i16 undef, 0
  br label %for.body2

for.body2:                                        ; preds = %if.end6, %for.cond1thread-pre-split
  br i1 %tobool3, label %if.end, label %for.end

if.end:                                           ; preds = %for.body2
  br i1 %tobool4, label %if.end6, label %for.cond1thread-pre-split

if.end6:                                          ; preds = %if.end
  br i1 undef, label %for.body2, label %for.end

for.end:                                          ; preds = %if.end6, %for.body2
  ret void
}
