; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

define i32 @main() {
; CHECK-LABEL: @main
; CHECK: %[[call:.*]] = call ptr @ctime(ptr null)
; CHECK: %[[cast:.*]] = ptrtoint ptr %[[call]] to i32
; CHECK: ret i32 %[[cast]]
entry:
  %tmp = call i32 @ctime( ptr null )          ; <i32> [#uses=1]
  ret i32 %tmp
}

declare ptr @ctime(ptr)

define internal { i8 } @foo(ptr) {
entry:
  ret { i8 } { i8 0 }
}

define void @test_struct_ret() {
; CHECK-LABEL: @test_struct_ret
; CHECK-NOT: bitcast
entry:
  %0 = call { i8 } @foo(ptr null)
  ret void
}

declare i32 @fn1(i32)

define i32 @test1(ptr %a) {
; CHECK-LABEL: @test1
; CHECK:      %[[cast:.*]] = ptrtoint ptr %a to i32
; CHECK-NEXT: %[[call:.*]] = tail call i32 @fn1(i32 %[[cast]])
; CHECK-NEXT: ret i32 %[[call]]
entry:
  %call = tail call i32 @fn1(ptr %a)
  ret i32 %call
}

declare i32 @fn2(i16)

define i32 @test2(ptr %a) {
; CHECK-LABEL: @test2
; CHECK:      %[[call:.*]] = tail call i32 @fn2(ptr %a)
; CHECK-NEXT: ret i32 %[[call]]
entry:
  %call = tail call i32 @fn2(ptr %a)
  ret i32 %call
}

declare i32 @fn3(i64)

define i32 @test3(ptr %a) {
; CHECK-LABEL: @test3
; CHECK:      %[[call:.*]] = tail call i32 @fn3(ptr %a)
; CHECK-NEXT: ret i32 %[[call]]
entry:
  %call = tail call i32 @fn3(ptr %a)
  ret i32 %call
}

declare i32 @fn4(i32) "thunk"

define i32 @test4(ptr %a) {
; CHECK-LABEL: @test4
; CHECK:      %[[call:.*]] = tail call i32 @fn4(ptr %a)
; CHECK-NEXT: ret i32 %[[call]]
entry:
  %call = tail call i32 @fn4(ptr %a)
  ret i32 %call
}

declare i1 @fn5(ptr byval({ i32, i32 }) align 4 %r)

define i1 @test5() {
; CHECK-LABEL: @test5
; CHECK:      %[[call:.*]] = call i1 @fn5(i32 {{.*}}, i32 {{.*}})
; CHECK-NEXT: ret i1 %[[call]]
  %1 = alloca { i32, i32 }, align 4
  %2 = getelementptr inbounds { i32, i32 }, ptr %1, i32 0, i32 0
  %3 = load i32, ptr %2, align 4
  %4 = getelementptr inbounds { i32, i32 }, ptr %1, i32 0, i32 1
  %5 = load i32, ptr %4, align 4
  %6 = call i1 @fn5(i32 %3, i32 %5)
  ret i1 %6
}
