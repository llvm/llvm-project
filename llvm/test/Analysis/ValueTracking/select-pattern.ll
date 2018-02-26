; RUN: opt -simplifycfg < %s -S | FileCheck %s

; The dead code would cause a select that had itself
; as an operand to be analyzed. This would then cause
; infinite recursion and eventual crash.

define void @PR36045(i1 %t, i32* %b) {
; CHECK-LABEL: @PR36045(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
;
entry:
  br i1 %t, label %if, label %end

if:
  br i1 %t, label %unreach, label %pre

unreach:
  unreachable

pre:
  %p = phi i32 [ 70, %if ], [ %sel, %for ]
  br label %for

for:
  %cmp = icmp sgt i32 %p, 8
  %add = add i32 %p, 2
  %sel = select i1 %cmp, i32 %p, i32 %add
  %cmp21 = icmp ult i32 %sel, 21
  br i1 %cmp21, label %pre, label %for.end

for.end:
  br i1 %t, label %unreach2, label %then12

then12:
  store i32 0, i32* %b
  br label %unreach2

unreach2:
  %spec = phi i32 [ %sel, %for.end ], [ 42, %then12 ]
  unreachable

end:
  ret void
}

