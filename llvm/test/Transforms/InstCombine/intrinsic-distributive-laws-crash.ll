; RUN: opt -passes=instcombine -disable-output %s
define i32 @j(i32 range(i32 8, 7) %b) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %k.0 = phi i64 [ -1910, %entry ], [ 995, %do.body ]
  %0 = trunc nsw i64 %k.0 to i32
  %conv1 = sub nsw i32 0, %0
  %sub14 = sub nsw i32 2, %0
  %add = add nsw i32 %0, 2
  %cmp.i = icmp slt i32 %conv1, %add
  %cond.i = call i32 @llvm.smin.i32(i32 %conv1, i32 %sub14)
  %cond5.i = select i1 %cmp.i, i32 0, i32 %cond.i
  %tobool.not = icmp eq i32 %b, 7
  br i1 %tobool.not, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  ret i32 %cond5.i
}
