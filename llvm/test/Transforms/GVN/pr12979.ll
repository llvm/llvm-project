; RUN: opt -passes=gvn -S < %s | FileCheck %s

define i32 @test1(i32 %x, i32 %y) {
; CHECK: @test1(i32 %x, i32 %y)
; CHECK: %add1 = add i32 %x, %y
; CHECK: %foo = add i32 %add1, %add1

  %add1 = add nsw i32 %x, %y
  %add2 = add     i32 %x, %y
  %foo = add i32 %add1, %add2
  ret i32 %foo
}

define i32 @test2(i32 %x, i32 %y) {
; CHECK: @test2(i32 %x, i32 %y)
; CHECK: %add1 = add i32 %x, %y
; CHECK: %foo = add i32 %add1, %add1

  %add1 = add nuw i32 %x, %y
  %add2 = add     i32 %x, %y
  %foo = add i32 %add1, %add2
  ret i32 %foo
}

define i32 @test3(i32 %x, i32 %y) {
; CHECK: @test3(i32 %x, i32 %y)
; CHECK: %add1 = add i32 %x, %y
; CHECK: %foo = add i32 %add1, %add1

  %add1 = add nuw nsw i32 %x, %y
  %add2 = add     i32 %x, %y
  %foo = add i32 %add1, %add2
  ret i32 %foo
}

define i32 @test4(i32 %x, i32 %y) {
; CHECK: @test4(i32 %x, i32 %y)
; CHECK: %add1 = add nsw i32 %x, %y
; CHECK: %foo = add i32 %add1, %add1

  %add1 = add nsw i32 %x, %y
  %add2 = add nsw i32 %x, %y
  %foo = add i32 %add1, %add2
  ret i32 %foo
}

define i32 @test5(i32 %x, i32 %y) {
; CHECK: @test5(i32 %x, i32 %y)
; CHECK: %add1 = add i32 %x, %y
; CHECK: %foo = add i32 %add1, %add1

  %add1 = add nuw i32 %x, %y
  %add2 = add nsw i32 %x, %y
  %foo = add i32 %add1, %add2
  ret i32 %foo
}

define i32 @test6(i32 %x, i32 %y) {
; CHECK: @test6(i32 %x, i32 %y)
; CHECK: %add1 = add nsw i32 %x, %y
; CHECK: %foo = add i32 %add1, %add1

  %add1 = add nuw nsw i32 %x, %y
  %add2 = add nsw i32 %x, %y
  %foo = add i32 %add1, %add2
  ret i32 %foo
}

define i32 @test7(i32 %x, i32 %y) {
; CHECK: @test7(i32 %x, i32 %y)
; CHECK: %add1 = add i32 %x, %y
; CHECK-NOT: what_is_this
; CHECK: %foo = add i32 %add1, %add1

  %add1 = add i32 %x, %y, !what_is_this !{}
  %add2 = add i32 %x, %y
  %foo = add i32 %add1, %add2
  ret i32 %foo
}

declare void @mumble(i2, i2)

define void @test8(i2 %x) {
; CHECK-LABEL: @test8(
; CHECK:      %[[ashr:.*]] = ashr i2 %x, 1
; CHECK-NEXT: call void @mumble(i2 %[[ashr]], i2 %[[ashr]])
; CHECK-NEXT: ret void

  %ashr0 = ashr exact i2 %x, 1
  %ashr1 = ashr i2 %x, 1
  call void @mumble(i2 %ashr0, i2 %ashr1)
  ret void
}
