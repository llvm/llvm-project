; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare i32 @f32(ptr, ptr)

declare i32 @f64(ptr, ptr)

define i1 @icmp_func() {
; CHECK-LABEL: @icmp_func(
; CHECK: ret i1 false
  %cmp = icmp eq ptr @f32, @f64
  ret i1 %cmp
}

define i1 @icmp_fptr(ptr) {
; CHECK-LABEL: @icmp_fptr(
; CHECK: %cmp = icmp ne ptr %0, @f32
; CHECK: ret i1 %cmp
  %cmp = icmp ne ptr @f32, %0
  ret i1 %cmp
}

@b = external global i32

define i32 @icmp_glob(i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @icmp_glob(i32 %x, i32 %y)
; CHECK-NEXT:   ret i32 %y
;
  %cmp = icmp eq ptr @icmp_glob, @b
  %sel = select i1 %cmp, i32 %x, i32 %y
  ret i32 %sel
}
