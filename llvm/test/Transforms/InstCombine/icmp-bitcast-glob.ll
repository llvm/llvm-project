; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare i32 @f32(i32**, i32**)

declare i32 @f64(i64**, i64**)

define i1 @icmp_func() {
; CHECK-LABEL: @icmp_func(
; CHECK: ret i1 false
  %cmp = icmp eq i32 (i8*, i8*)* bitcast (i32 (i32**, i32**)* @f32 to i32 (i8*, i8*)*), bitcast (i32 (i64**, i64**)* @f64 to i32 (i8*, i8*)*)
  ret i1 %cmp
}

define i1 @icmp_fptr(i32 (i8*, i8*)*) {
; CHECK-LABEL: @icmp_fptr(
; CHECK: %cmp = icmp ne i32 (i8*, i8*)* %0, bitcast (i32 (i32**, i32**)* @f32 to i32 (i8*, i8*)*)
; CHECK: ret i1 %cmp
  %cmp = icmp ne i32 (i8*, i8*)* bitcast (i32 (i32**, i32**)* @f32 to i32 (i8*, i8*)*), %0
  ret i1 %cmp
}

@b = external global i32

define i32 @icmp_glob(i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @icmp_glob(i32 %x, i32 %y)
; CHECK-NEXT:   ret i32 %y
;
  %sel = select i1 icmp eq (i32* bitcast (i32 (i32, i32)* @icmp_glob to i32*), i32* @b), i32 %x, i32 %y
  ret i32 %sel
}
