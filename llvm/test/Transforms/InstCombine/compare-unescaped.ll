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

define i1 @compare_distinct_mallocs() {
  %m = call i8* @malloc(i64 4)
  %n = call i8* @malloc(i64 4)
  %cmp = icmp eq i8* %m, %n
  ret i1 %cmp
  ; CHECK-LABEL: compare_distinct_mallocs
  ; CHECK: ret i1 false
}

; the compare is folded to true since the folding compare looks through bitcasts. 
; call to malloc and the bitcast instructions are elided after that since there are no uses of the malloc 
define i1 @compare_samepointer_under_bitcast() {
  %m = call i8* @malloc(i64 4)
  %bc = bitcast i8* %m to i32*
  %bcback = bitcast i32* %bc to i8*
  %cmp = icmp eq i8* %m, %bcback
  ret i1 %cmp
; CHECK-LABEL: compare_samepointer_under_bitcast
; CHECK: ret i1 true 
}

; the compare is folded to true since the folding compare looks through bitcasts. 
; call to malloc and the bitcast instructions are elided after that since there are no uses of the malloc 
define i1 @compare_samepointer_escaped() {
  %m = call i8* @malloc(i64 4)
  %bc = bitcast i8* %m to i32*
  %bcback = bitcast i32* %bc to i8*
  %cmp = icmp eq i8* %m, %bcback
  call void @f() [ "deopt"(i8* %m) ]
  ret i1 %cmp
; CHECK-LABEL: compare_samepointer_escaped
; CHECK-NEXT: %m = call i8* @malloc(i64 4)
; CHECK-NEXT: call void @f() [ "deopt"(i8* %m) ]
; CHECK: ret i1 true 
}

; The malloc call for %m cannot be elided since it is used in the call to function f.
; However, the cmp can be folded to true as %n doesnt escape and %m, %n are distinct allocations
define i1 @compare_distinct_pointer_escape() {
  %m = call i8* @malloc(i64 4)
  %n = call i8* @malloc(i64 4)
  tail call void @f() [ "deopt"(i8* %m) ]
  %cmp = icmp ne i8* %m, %n
  ret i1 %cmp
; CHECK-LABEL: compare_distinct_pointer_escape
; CHECK-NEXT: %m = call i8* @malloc(i64 4)
; CHECK-NEXT: tail call void @f() [ "deopt"(i8* %m) ]
; CHECK-NEXT: ret i1 true
}
