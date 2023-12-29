; RUN: opt --passes=indvars -disable-output < %s

target datalayout = "e-n32:64"

declare void @g(i64)

define void @f(ptr %block_address_arg)  {

; This loop isn't in Simplified Form. Ensure that there's no SCEV
; expansion, which would cause a crash.
f_entry:
  indirectbr ptr %block_address_arg, [label %f.exit, label %f_for.cond]

f_for.cond:                       ; preds = %f_for.body, %f_entry
  %i = phi i32 [ %j, %f_for.body ], [ 0, %f_entry ]
  %cmp = icmp ult i32 %i, 64
  br i1 %cmp, label %f_for.body, label %f_for.end

f_for.body:                       ; preds = %f_for.cond
  %j = add nuw nsw i32 %i, 1
  br label %f_for.cond

; Indvars pass visits this loop, since it's in Simplified Form.
; Because of the integer extension in that loop's body, SCEV expander may also
; visit the first loop.
f_for.end:                        ; preds = %f_for.cond
  %k = phi i32 [ %i, %f_for.cond ]
  br label %f_for2.body

f_for2.body:                       ; preds = %f_for2.body, %f_for.end
  %l = phi i32 [ %k, %f_for.end ], [ %n, %f_for2.body ]
  %m = zext i32 %l to i64
  call void @g(i64 %m)
  %n = add nuw nsw i32 %l, 1
  br label %f_for2.body

f.exit:                                ; preds = %f_entry
  ret void
}
