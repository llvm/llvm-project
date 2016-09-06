; RUN: opt < %s -inline -sroa -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define i32 @test1f(i32 %i) {
        ret i32 %i
}

define i32 @test1(i32 %W) {
        %X = call i32 @test1f(i32 7)
        %Y = add i32 %X, %W
        ret i32 %Y
; CHECK-LABEL: @test1(
; CHECK-NEXT: %Y = add i32 7, %W
; CHECK-NEXT: ret i32 %Y
}



; rdar://7339069

%T = type { i32, i32 }

; CHECK-NOT: @test2f(
define internal %T* @test2f(i1 %cond, %T* %P) {
  br i1 %cond, label %T, label %F
  
T:
  %A = getelementptr %T, %T* %P, i32 0, i32 0
  store i32 42, i32* %A
  ret %T* %P
  
F:
  ret %T* %P
}

define i32 @test2(i1 %cond) {
  %A = alloca %T
  
  %B = call %T* @test2f(i1 %cond, %T* %A)
  %C = getelementptr %T, %T* %B, i32 0, i32 0
  %D = load i32, i32* %C
  ret i32 %D
  
; CHECK-LABEL: @test2(
; CHECK-NOT: = alloca
; CHECK: ret i32
}

declare void @barrier() noduplicate

define internal i32 @f() {
  call void @barrier() noduplicate
  ret i32 1
}

define i32 @g() {
  call void @barrier() noduplicate
  ret i32 2
}

define internal i32 @h() {
  call void @barrier() noduplicate
  ret i32 3
}

define i32 @test3() {
  %b = call i32 @f()
  ret i32 %b
}

; The call to @f cannot be inlined as there is another callsite
; calling @f, and @f contains a noduplicate call.
;
; The call to @g cannot be inlined as it has external linkage.
;
; The call to @h *can* be inlined.

; CHECK-LABEL: @test(
define i32 @test() {
; CHECK: call i32 @f()
  %a = call i32 @f()
; CHECK: call i32 @g()
  %b = call i32 @g()
; CHECK-NOT: call i32 @h()
  %c = call i32 @h()

  %d = add i32 %a, %b
  %e = add i32 %d, %c

  ret i32 %e
; CHECK: }
}
