; RUN: opt -instcombine -S < %s | FileCheck %s

@gp = global i32* null, align 8

declare i8* @malloc(i64) #1

define i1 @compare_global_trivialeq() {
  %m = call i8* @malloc(i64 4)
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp, align 8
  %cmp = icmp eq i32* %bc, %lgp
  ret i1 %cmp
; CHECK-LABEL: compare_global_trivialeq
; CHECK: ret i1 false
}

define i1 @compare_global_trivialne() {
  %m = call i8* @malloc(i64 4)
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp, align 8
  %cmp = icmp ne i32* %bc, %lgp
  ret i1 %cmp
; CHECK-LABEL: compare_global_trivialne
; CHECK: ret i1 true 
}


; Although the %m is marked nocapture in the deopt operand in call to function f,
; we cannot remove the alloc site: call to malloc
; FIXME: The comparison should fold to false irrespective of whether the call to malloc can be elided or not
declare void @f()
define i32 @compare_and_call_with_deopt() {
; CHECK-LABEL: compare_and_call_with_deopt
  %m = call i8* @malloc(i64 24)
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp, align 8
  %cmp = icmp eq i32* %bc, %lgp
  %rt = zext i1 %cmp to i32
  tail call void @f() [ "deopt"(i8* %m) ]
  ret i32 %rt 
; CHECK: ret i32 %rt
}
