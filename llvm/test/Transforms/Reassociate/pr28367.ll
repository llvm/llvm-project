; RUN: opt < %s -reassociate -S

; PR28367

; Check to make sure this test does not assert or segfault.  If we get too
; aggressive with retrying instructions it's possible to invalidate our
; iterator.  See PR28367 for complete details.

define void @fn1(i32 %a, i1 %c, i32* %ptr)  {
entry:
  br label %for.cond

for.cond:
  %d.0 = phi i32 [ 1, %entry ], [ 2, %for.body ]
  br i1 %c, label %for.end, label %for.body

for.body:
  %sub1 = sub i32 %a, %d.0
  %dead1 = add i32 %sub1, 1
  %dead2 = mul i32 %dead1, 3
  %dead3 = mul i32 %dead2, %sub1
  %sub2 = sub nsw i32 0, %d.0
  store i32 %sub2, i32* %ptr, align 4
  br label %for.cond

for.end:
  ret void
}
