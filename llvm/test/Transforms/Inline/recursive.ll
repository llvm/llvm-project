; RUN: opt -inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

; rdar://10853263

; Make sure that the callee is still here.
; CHECK-LABEL: define i32 @callee(
define i32 @callee(i32 %param) {
 %yyy = alloca [100000 x i8]
 %r = bitcast [100000 x i8]* %yyy to i8*
 call void @foo2(i8* %r)
 ret i32 4
}

; CHECK-LABEL: define i32 @caller(
; CHECK-NEXT: entry:
; CHECK-NOT: alloca
; CHECK: ret
define i32 @caller(i32 %param) {
entry:
  %t = call i32 @foo(i32 %param)
  %cmp = icmp eq i32 %t, -1
  br i1 %cmp, label %exit, label %cont

cont:
  %r = call i32 @caller(i32 %t)
  %f = call i32 @callee(i32 %r)
  br label %cont
exit:
  ret i32 4
}

declare void @foo2(i8* %in)

declare i32 @foo(i32 %param)

