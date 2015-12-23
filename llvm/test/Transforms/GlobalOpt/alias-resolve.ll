; RUN: opt < %s -globalopt -S | FileCheck %s

@foo1 = alias void (), void ()* @foo2
; CHECK: @foo1 = alias void (), void ()* @bar2

@foo2 = alias void(), void()* @bar1
; CHECK: @foo2 = alias void (), void ()* @bar2

@bar1  = alias void (), void ()* @bar2
; CHECK: @bar1 = alias void (), void ()* @bar2

@weak1 = weak alias void (), void ()* @bar2
; CHECK: @weak1 = weak alias void (), void ()* @bar2

@bar4 = private unnamed_addr constant [2 x i8*] zeroinitializer
@foo4 = weak_odr unnamed_addr alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* @bar4, i32 0, i32 1)
; CHECK: @foo4 = weak_odr unnamed_addr alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* @bar4, i32 0, i32 1)

define void @bar2() {
  ret void
}
; CHECK: define void @bar2()

define void @baz() {
entry:
         call void @foo1()
; CHECK: call void @bar2()

         call void @foo2()
; CHECK: call void @bar2()

         call void @bar1()
; CHECK: call void @bar2()

         call void @weak1()
; CHECK: call void @weak1()
         ret void
}

@foo3 = alias void (), void ()* @bar3
; CHECK-NOT: bar3

define internal void @bar3() {
  ret void
}
;CHECK: define void @foo3
