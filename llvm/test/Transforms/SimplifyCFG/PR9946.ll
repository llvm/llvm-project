; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output < %s

@foo = external constant i32

define i32 @f() {
entry:
  %and = and i64 ptrtoint (ptr @foo to i64), 15
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  br label %return

return:                                           ; preds = %if.end, %if.then
  %storemerge = phi i32 [ 1, %if.end ], [ 0, %if.then ]
  ret i32 %storemerge
}
