; RUN: opt < %s -loop-reduce -S | FileCheck %s

; Provide legal integer types.
target datalayout = "n8:16:32:64"

@d = common global i32 0, align 4

define void @fn2(i32 %x) nounwind uwtable {
entry:
  br label %for.cond

for.cond:
  %g.0 = phi i32 [ 0, %entry ], [ %dec, %for.cond ]
  %tobool = icmp eq i32 %x, 0
  %dec = add nsw i32 %g.0, -1
  br i1 %tobool, label %for.cond, label %for.end

for.end:
; CHECK:  %tmp1 = load i32, i32* @d, align 4
; CHECK-NEXT:  %tmp2 = load i32, i32* @d, align 4
; CHECK-NEXT:  %0 = sub i32 %tmp1, %tmp2

  %tmp1 = load i32, i32* @d, align 4
  %add = add nsw i32 %tmp1, %g.0
  %tmp2 = load i32, i32* @d, align 4
  %tobool26 = icmp eq i32 %x, 0
  br i1 %tobool26, label %for.end5, label %for.body.lr.ph

for.body.lr.ph:
  %tobool3 = icmp ne i32 %tmp2, %add
  br label %for.end5

for.end5:
  ret void
}


