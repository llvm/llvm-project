; Test stack pointer restore after setjmp() with the function-call safestack ABI.
; RUN: opt -safe-stack -S -mtriple=arm-linux-androideabi < %s -o - | FileCheck %s

@env = global [64 x i32] zeroinitializer, align 4

define void @f(i32 %b) safestack {
entry:
; CHECK: %[[SPA:.*]] = call i8** @__safestack_pointer_address()
; CHECK: %[[USP:.*]] = load i8*, i8** %[[SPA]]
; CHECK: %[[USDP:.*]] = alloca i8*
; CHECK: store i8* %[[USP]], i8** %[[USDP]]
; CHECK: call i32 @setjmp

  %call = call i32 @setjmp(i32* getelementptr inbounds ([64 x i32], [64 x i32]* @env, i32 0, i32 0)) returns_twice

; CHECK: %[[USP2:.*]] = load i8*, i8** %[[USDP]]
; CHECK: store i8* %[[USP2]], i8** %[[SPA]]

  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %0 = alloca [42 x i8], align 1
  %.sub = getelementptr inbounds [42 x i8], [42 x i8]* %0, i32 0, i32 0
  call void @_Z7CapturePv(i8* %.sub)
  br label %if.end

if.end:
; CHECK: store i8* %[[USP:.*]], i8** %[[SPA:.*]]

  ret void
}

declare i32 @setjmp(i32*) returns_twice

declare void @_Z7CapturePv(i8*)
