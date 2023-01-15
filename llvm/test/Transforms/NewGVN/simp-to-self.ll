; RUN: opt -S < %s -passes=newgvn | FileCheck %s

; CHECK-LABEL: for.cond:
; CHECK-NEXT:    %lv = load i32, ptr @a
; CHECK-NEXT:    %bf.clear = and i32 %lv, -131072
; CHECK-NEXT:    %bf.set = or i32 1, %bf.clear
; CHECK-NEXT:    br i1 %bc, label %for.cond, label %exit
@a = external global i64

define void @fn1(i1 %bc) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond1.1, %entry
  %tmp = phi i1 [ undef, %entry ], [ 1, %for.cond ]
  %conv = zext i1 %tmp to i32
  %lv = load i32, ptr @a
  %bf.clear = and i32 %lv, -131072
  %bf.set = or i32 %conv, %bf.clear
  %bf.clear.1 = and i32 %bf.set, -131072
  %bf.set.1 = or i32 1, %bf.clear.1
  br i1 %bc, label %for.cond, label %exit

exit:                              ; preds = %for.cond1
  store i32 %bf.set.1, ptr @a
  ret void
}
