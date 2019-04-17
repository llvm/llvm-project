; RUN: opt -simplifycfg -disable-output < %s

@foo = external constant i32

define i32 @f() {
entry:
  br i1 icmp eq (i64 and (i64 ptrtoint (i32* @foo to i64), i64 15), i64 0), label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  br label %return

return:                                           ; preds = %if.end, %if.then
  %storemerge = phi i32 [ 1, %if.end ], [ 0, %if.then ]
  ret i32 %storemerge
}
