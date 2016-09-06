; RUN: opt -mem2reg < %s -S | FileCheck %s
; RUN: opt -passes=mem2reg < %s -S | FileCheck %s

declare i32 @def(i32)
declare i1 @use(i32)

; Special case of a single-BB alloca does not apply here since the load
; is affected by the following store. Expect this case to be identified
; and a PHI node to be created.
define void @test1() {
; CHECK-LABEL: @test1(
 entry:
  %t = alloca i32
  br label %loop

 loop:
  %v = load i32, i32* %t
  %c = call i1 @use(i32 %v)
; CHECK: [[PHI:%.*]] = phi i32 [ undef, %entry ], [ %n, %loop ]
; CHECK: call i1 @use(i32 [[PHI]])
  %n = call i32 @def(i32 7)
  store i32 %n, i32* %t
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

; Same as above, except there is no following store. The alloca should just be
; replaced with an undef
define void @test2() {
; CHECK-LABEL: @test2(
 entry:
  %t = alloca i32
  br label %loop

 loop:
  %v = load i32, i32* %t
  %c = call i1 @use(i32 %v)
; CHECK: %c = call i1 @use(i32 undef)
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}
