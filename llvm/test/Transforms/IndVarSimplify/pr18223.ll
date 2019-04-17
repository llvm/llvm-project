; RUN: opt -indvars -S < %s | FileCheck %s

; indvars should transform the phi node pair from the for-loop
; CHECK-LABEL: @main(
; CHECK: ret = phi i32 [ 0, %entry ], [ 0, {{.*}} ]

@c = common global i32 0, align 4

define i32 @main() #0 {
entry:
  %0 = load i32, i32* @c, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.body, label %exit

for.body:
  %inc2 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %sub = add i32 %inc2, -1
  %cmp1 = icmp uge i32 %sub, %inc2
  %conv = zext i1 %cmp1 to i32
  br label %for.inc

for.inc:
  %inc = add nsw i32 %inc2, 1
  %cmp = icmp slt i32 %inc, 5
  br i1 %cmp, label %for.body, label %exit

exit:
  %ret = phi i32 [ 0, %entry ], [ %conv, %for.inc ]
  ret i32 %ret
}
