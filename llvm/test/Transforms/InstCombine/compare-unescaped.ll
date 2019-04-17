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
; The comparison should fold to false irrespective of whether the call to malloc can be elided or not
declare void @f()
define i1 @compare_and_call_with_deopt() {
; CHECK-LABEL: compare_and_call_with_deopt
  %m = call i8* @malloc(i64 24)
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp, align 8, !nonnull !0
  %cmp = icmp eq i32* %lgp, %bc
  tail call void @f() [ "deopt"(i8* %m) ]
  ret i1 %cmp
; CHECK: ret i1 false
}

; Same functon as above with deopt operand in function f, but comparison is NE
define i1 @compare_ne_and_call_with_deopt() {
; CHECK-LABEL: compare_ne_and_call_with_deopt
  %m = call i8* @malloc(i64 24)
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp, align 8, !nonnull !0
  %cmp = icmp ne i32* %lgp, %bc
  tail call void @f() [ "deopt"(i8* %m) ]
  ret i1 %cmp
; CHECK: ret i1 true
}

; Same function as above, but global not marked nonnull, and we cannot fold the comparison
define i1 @compare_ne_global_maybe_null() {
; CHECK-LABEL: compare_ne_global_maybe_null
  %m = call i8* @malloc(i64 24)
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp
  %cmp = icmp ne i32* %lgp, %bc
  tail call void @f() [ "deopt"(i8* %m) ]
  ret i1 %cmp
; CHECK: ret i1 %cmp
}

; FIXME: The comparison should fold to false since %m escapes (call to function escape)
; after the comparison.
declare void @escape(i8*)
define i1 @compare_and_call_after() {
; CHECK-LABEL: compare_and_call_after
  %m = call i8* @malloc(i64 24)
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp, align 8, !nonnull !0
  %cmp = icmp eq i32* %bc, %lgp
  br i1 %cmp, label %escape_call, label %just_return

escape_call:
 call void @escape(i8* %m)
 ret i1 true

just_return:
 ret i1 %cmp
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
; The malloc call for %m cannot be elided since it is used in the call to function f.
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

; Technically, we can fold the %cmp2 comparison, even though %m escapes through
; the ret statement since `ret` terminates the function and we cannot reach from
; the ret to cmp. 
; FIXME: Folding this %cmp2 when %m escapes through ret could be an issue with
; cross-threading data dependencies since we do not make the distinction between
; atomic and non-atomic loads in capture tracking.
define i8* @compare_ret_escape(i8* %c) {
  %m = call i8* @malloc(i64 4)
  %n = call i8* @malloc(i64 4)
  %cmp = icmp eq i8* %n, %c
  br i1 %cmp, label %retst, label %chk

retst:
  ret i8* %m

chk:
  %bc = bitcast i8* %m to i32*
  %lgp = load i32*, i32** @gp, align 8, !nonnull !0
  %cmp2 = icmp eq i32* %bc, %lgp
  br i1 %cmp2, label %retst,  label %chk2

chk2:
  ret i8* %n
; CHECK-LABEL: compare_ret_escape
; CHECK: %cmp = icmp eq i8* %n, %c
; CHECK: %cmp2 = icmp eq i32* %lgp, %bc
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

!0 = !{}
