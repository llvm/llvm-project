; Test stack pointer restore after setjmp() with the function-call safestack ABI.
; RUN: opt -safe-stack -S -mtriple=arm-linux-androideabi < %s -o - | FileCheck %s

@env = global [64 x i32] zeroinitializer, align 4

define void @f(i32 %b) safestack {
entry:
; CHECK: %[[SPA:.*]] = call ptr @__safestack_pointer_address()
; CHECK: %[[USP:.*]] = load ptr, ptr %[[SPA]]
; CHECK: %[[USDP:.*]] = alloca ptr
; CHECK: store ptr %[[USP]], ptr %[[USDP]]
; CHECK: call i32 @setjmp

  %call = call i32 @setjmp(ptr @env) returns_twice

; CHECK: %[[USP2:.*]] = load ptr, ptr %[[USDP]]
; CHECK: store ptr %[[USP2]], ptr %[[SPA]]

  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %0 = alloca [42 x i8], align 1
  call void @_Z7CapturePv(ptr %0)
  br label %if.end

if.end:
; CHECK: store ptr %[[USP:.*]], ptr %[[SPA:.*]]

  ret void
}

declare i32 @setjmp(ptr) returns_twice

declare void @_Z7CapturePv(ptr)
