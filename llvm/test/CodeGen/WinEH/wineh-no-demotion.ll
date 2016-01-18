; RUN: opt -mtriple=x86_x64-pc-windows-msvc -S -winehprepare -disable-demotion < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare void @f()

declare i32 @g()

declare void @h(i32)

; CHECK-LABEL: @test1(
define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
          to label %invoke.cont1 unwind label %left

invoke.cont1:
  invoke void @f()
          to label %invoke.cont2 unwind label %right

invoke.cont2:
  invoke void @f()
          to label %exit unwind label %inner

left:
  %0 = cleanuppad []
  br label %shared

right:
  %1 = cleanuppad []
  br label %shared

shared:
  %x = call i32 @g()
  invoke void @f()
          to label %shared.cont unwind label %inner

shared.cont:
  unreachable

inner:
  %phi = phi i32 [ %x, %shared ], [ 0, %invoke.cont2 ]
  %i = cleanuppad []
  call void @h(i32 %phi)
  unreachable

; CHECK [[INNER_INVOKE_CONT2:inner.*]]:
  ; CHECK: call void @h(i32 0)

; CHECK [[INNER_RIGHT:inner.*]]:
  ; CHECK: call void @h(i32 %x)

; CHECK [[INNER_LEFT:inner.*]]:
  ; CHECK: call void @h(i32 %x.for.left)

exit:
  unreachable
}

; CHECK-LABEL: @test2(
define void @test2() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
          to label %invoke.cont unwind label %left

invoke.cont:
  invoke void @f()
          to label %exit unwind label %right

left:
  cleanuppad []
  br label %shared

right:
  cleanuppad []
  br label %shared

shared:
  %x = call i32 @g()
  invoke void @f()
          to label %shared.cont unwind label %inner

shared.cont:
  unreachable

inner:
  %i = cleanuppad []
  call void @h(i32 %x)
  unreachable

; CHECK [[INNER_RIGHT:inner.*]]:
  ; CHECK: call void @h(i32 %x)

; CHECK [[INNER_LEFT:inner.*]]:
  ; CHECK: call void @h(i32 %x.for.left)

exit:
  unreachable
}

; CHECK-LABEL: @test3(
define void @test3() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
          to label %invoke.cont unwind label %terminate

invoke.cont:
  ret void

terminate:
; CHECK:  cleanuppad []
; CHECK:  call void @__std_terminate()
; CHECK:  unreachable
  terminatepad [void ()* @__std_terminate] unwind to caller
}

declare void @__std_terminate()
