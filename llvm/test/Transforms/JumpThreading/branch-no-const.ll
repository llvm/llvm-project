; RUN: opt < %s -jump-threading -S | not grep phi

declare i8 @mcguffin()

define i32 @test(i1 %foo, i8 %b) {
entry:
  %a = call i8 @mcguffin()
  br i1 %foo, label %bb1, label %bb2
bb1:
  br label %jt
bb2:
  br label %jt
jt:
  %x = phi i8 [%a, %bb1], [%b, %bb2]
  %A = icmp eq i8 %x, %a
  br i1 %A, label %rt, label %rf
rt:
  ret i32 7
rf:
  ret i32 8
}
