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

; Check that when inlining a non-recursive path into a function's own body that
; we get the re-mapping of instructions correct.
define i32 @test_recursive_inlining_remapping(i1 %init, i8* %addr) {
; CHECK-LABEL: define i32 @test_recursive_inlining_remapping(
bb:
  %n = alloca i32
  br i1 %init, label %store, label %load
; CHECK-NOT:     alloca
;
; CHECK:         %[[N:.*]] = alloca i32
; CHECK-NEXT:    br i1 %init,

store:
  store i32 0, i32* %n
  %cast = bitcast i32* %n to i8*
  %v = call i32 @test_recursive_inlining_remapping(i1 false, i8* %cast)
  ret i32 %v
; CHECK-NOT:     call
;
; CHECK:         store i32 0, i32* %[[N]]
; CHECK-NEXT:    %[[CAST:.*]] = bitcast i32* %[[N]] to i8*
; CHECK-NEXT:    %[[INLINED_LOAD:.*]] = load i32, i32* %[[N]]
; CHECK-NEXT:    ret i32 %[[INLINED_LOAD]]
;
; CHECK-NOT:     call

load:
  %castback = bitcast i8* %addr to i32*
  %n.load = load i32, i32* %castback
  ret i32 %n.load
}
