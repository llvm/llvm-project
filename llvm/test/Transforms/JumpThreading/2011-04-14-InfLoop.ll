; RUN: opt -jump-threading < %s
; <rdar://problem/9284786>

%0 = type <{ i64, i16, i64, i8, i8 }>

@g_338 = external global %0, align 8

define void @func_1() nounwind ssp {
entry:
  ret void

for.cond1177:
  %inc1187 = add nsw i32 0, 1
  %cmp1179 = icmp slt i32 %inc1187, 5
  br i1 %cmp1179, label %for.cond1177, label %land.rhs1320

land.rhs1320:
  %tmp1324 = load volatile i64, i64* getelementptr inbounds (%0, %0* @g_338, i64 0, i32 2), align 1
  br label %if.end.i

if.end.i:
  %tobool.pr.i = phi i1 [ false, %if.end.i ], [ false, %land.rhs1320 ]
  br i1 %tobool.pr.i, label %return, label %if.end.i

return:
  ret void
}
