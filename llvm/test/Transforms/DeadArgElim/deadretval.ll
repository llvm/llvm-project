; RUN: opt < %s -deadargelim -S | FileCheck %s

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

; CHECK-LABEL: define internal i8* @callee4()

define internal i8* @callee4(i8* %a0) {
  ret i8* @g0;
}

declare void @llvm.objc.clang.arc.noop.use(...)

; CHECK-LABEL: define i8* @test4(
; CHECK: tail call i8* @callee4() [ "clang.arc.attachedcall"(i64 0) ]

define i8* @test4() {
  %call = tail call i8* @callee4(i8* @g0) [ "clang.arc.attachedcall"(i64 0) ]
  call void (...) @llvm.objc.clang.arc.noop.use(i8* %call)
  ret i8* @g0
}
