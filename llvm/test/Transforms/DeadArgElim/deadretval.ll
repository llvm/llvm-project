; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

@g0 = global i8 0, align 8

; CHECK-NOT: DEAD

; Dead arg only used by dead retval
define internal i32 @test(i32 %DEADARG) {
        ret i32 %DEADARG
}

define i32 @test2(i32 %A) {
        %DEAD = call i32 @test( i32 %A )                ; <i32> [#uses=0]
        ret i32 123
}

define i32 @test3() {
        %X = call i32 @test2( i32 3232 )                ; <i32> [#uses=1]
        %Y = add i32 %X, -123           ; <i32> [#uses=1]
        ret i32 %Y
}

; The callee function's return type shouldn't be changed if the call result is
; used.

; CHECK-LABEL: define internal ptr @callee4()

define internal ptr @callee4(ptr %a0) {
  ret ptr @g0;
}

declare void @llvm.objc.clang.arc.noop.use(...)

; CHECK-LABEL: define ptr @test4(
; CHECK: tail call ptr @callee4() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]

define ptr @test4() {
  %call = tail call ptr @callee4(ptr @g0) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %call)
  ret ptr @g0
}

declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
