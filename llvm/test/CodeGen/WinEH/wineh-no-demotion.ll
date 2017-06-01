; RUN: opt -mtriple=x86_x64-pc-windows-msvc -S -winehprepare -disable-demotion -disable-cleanups < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare i32 @__C_specific_handler(...)

declare void @f()

declare i32 @g()

declare void @h(i32)

; CHECK-LABEL: @test1(
define void @test1(i1 %bool) personality i32 (...)* @__C_specific_handler {
entry:
  invoke void @f()
          to label %invoke.cont unwind label %left

invoke.cont:
  invoke void @f()
          to label %exit unwind label %inner

left:
  %0 = cleanuppad within none []
  br i1 %bool, label %shared, label %cleanupret

cleanupret:
  cleanupret from %0 unwind label %right

right:
  %1 = cleanuppad within none []
  br label %shared

shared:
  %x = call i32 @g()
  invoke void @f()
          to label %shared.cont unwind label %inner

shared.cont:
  unreachable

inner:
  %phi = phi i32 [ %x, %shared ], [ 0, %invoke.cont ]
  %i = cleanuppad within none []
  call void @h(i32 %phi)
  unreachable

; CHECK: %phi = phi i32 [ %x, %shared ], [ 0, %invoke.cont ], [ %x.for.left, %shared.for.left ]
; CHECK: %i = cleanuppad within none []
; CHECK: call void @h(i32 %phi)

exit:
  unreachable
}

; CHECK-LABEL: @test2(
define void @test2(i1 %bool) personality i32 (...)* @__C_specific_handler {
entry:
  invoke void @f()
          to label %shared.cont unwind label %left

left:
  %0 = cleanuppad within none []
  br i1 %bool, label %shared, label %cleanupret

cleanupret:
  cleanupret from %0 unwind label %right

right:
  %1 = cleanuppad within none []
  br label %shared

shared:
  %x = call i32 @g()
  invoke void @f()
          to label %shared.cont unwind label %inner

shared.cont:
  unreachable

inner:
  %i = cleanuppad within none []
  call void @h(i32 %x)
  unreachable

; CHECK: %x1 = phi i32 [ %x.for.left, %shared.for.left ], [ %x, %shared ]
; CHECK: %i = cleanuppad within none []
; CHECK: call void @h(i32 %x1)

exit:
  unreachable
}

; CHECK-LABEL: @test4(
define void @test4(i1 %x) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
          to label %invoke.cont1 unwind label %left

invoke.cont1:
  invoke void @f()
          to label %exit unwind label %right

left:
  %0 = cleanuppad within none []
  br label %shared

right:
  %1 = cleanuppad within none []
  br i1 %x, label %shared, label %right.other

right.other:
  br label %shared

shared:
  %phi = phi i32 [ 1, %left ], [ 0, %right ], [ -1, %right.other ]
  call void @h(i32 %phi)
  unreachable

; CHECK: %phi = phi i32 [ 0, %right ], [ -1, %right.other ]
; CHECK: call void @h(i32 %phi)

; CHECK: %phi.for.left = phi i32 [ 1, %left ]
; CHECK: call void @h(i32 %phi.for.left)

exit:
  unreachable
}

declare void @__std_terminate()
