; RUN: opt < %s -S -loop-unroll -pass-remarks=loop-unroll -pass-remarks-with-hotness -unroll-count=16 2>&1 | FileCheck -check-prefix=COMPLETE-UNROLL %s
; RUN: opt < %s -S -loop-unroll -pass-remarks=loop-unroll -pass-remarks-with-hotness -unroll-count=4 2>&1 | FileCheck -check-prefix=PARTIAL-UNROLL %s

; COMPLETE-UNROLL: remark: {{.*}}: completely unrolled loop with 16 iterations (hotness: 300)
; PARTIAL-UNROLL: remark: {{.*}}: unrolled loop by a factor of 4 (hotness: 300)

define i32 @sum() !prof !0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %s.06 = phi i32 [ 0, %entry ], [ %add1, %for.body ]
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %i.05, 4
  %call = tail call i32 @baz(i32 %add) #2
  %add1 = add nsw i32 %call, %s.06
  %inc = add nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 16
  br i1 %exitcond, label %for.end, label %for.body, !prof !1

for.end:                                          ; preds = %for.body
  ret i32 %add1
}

declare i32 @baz(i32)

!0 = !{!"function_entry_count", i64 3}
!1 = !{!"branch_weights", i32 1, i32 99}
