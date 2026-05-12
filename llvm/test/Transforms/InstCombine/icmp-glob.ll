; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define i32 @f32(ptr, ptr) {
  ret i32 32
}

define i32 @f64(ptr, ptr) {
  ret i32 64
}

declare void @fdecl()

define i1 @icmp_func() {
; CHECK-LABEL: @icmp_func(
; CHECK: ret i1 false
  %cmp = icmp eq ptr @f32, @f64
  ret i1 %cmp
}

define i1 @icmp_func_decl() {
; CHECK-LABEL: @icmp_func_decl(
; CHECK: %cmp = icmp eq ptr @f32, @fdecl
; CHECK: ret i1 %cmp
  %cmp = icmp eq ptr @f32, @fdecl
  ret i1 %cmp
}

define i1 @icmp_fptr(ptr) {
; CHECK-LABEL: @icmp_fptr(
; CHECK: %cmp = icmp ne ptr %0, @f32
; CHECK: ret i1 %cmp
  %cmp = icmp ne ptr @f32, %0
  ret i1 %cmp
}

@b = global i32 1
@vdecl = external global i32

define i32 @icmp_glob(i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @icmp_glob(i32 %x, i32 %y)
; CHECK-NEXT:   ret i32 %y
;
  %cmp = icmp eq ptr @icmp_glob, @b
  %sel = select i1 %cmp, i32 %x, i32 %y
  ret i32 %sel
}

define i32 @icmp_glob_decl(i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @icmp_glob_decl(i32 %x, i32 %y)
; CHECK-NEXT:   %cmp = icmp eq ptr @icmp_glob, @vdecl
; CHECK-NEXT:   %sel = select i1 %cmp, i32 %x, i32 %y
; CHECK-NEXT:   ret i32 %sel
;
  %cmp = icmp eq ptr @icmp_glob, @vdecl
  %sel = select i1 %cmp, i32 %x, i32 %y
  ret i32 %sel
}
