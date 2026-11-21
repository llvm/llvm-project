; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK-LABEL: define i1 @udiv_i1(
; CHECK: %r = call i1 @llvm.masked.udiv.i1(i1 %a, i1 %b, i1 %m)
define i1 @udiv_i1(i1 %a, i1 %b, i1 %m) {
  %r = call i1 @llvm.masked.udiv.i1(i1 %a, i1 %b, i1 %m)
  ret i1 %r
}

; CHECK-LABEL: define i8 @sdiv_i8(
; CHECK: %r = call i8 @llvm.masked.sdiv.i8(i8 %a, i8 %b, i1 %m)
define i8 @sdiv_i8(i8 %a, i8 %b, i1 %m) {
  %r = call i8 @llvm.masked.sdiv.i8(i8 %a, i8 %b, i1 %m)
  ret i8 %r
}

; CHECK-LABEL: define i32 @urem_i32(
; CHECK: %r = call i32 @llvm.masked.urem.i32(i32 %a, i32 %b, i1 %m)
define i32 @urem_i32(i32 %a, i32 %b, i1 %m) {
  %r = call i32 @llvm.masked.urem.i32(i32 %a, i32 %b, i1 %m)
  ret i32 %r
}

; CHECK-LABEL: define i64 @srem_i64(
; CHECK: %r = call i64 @llvm.masked.srem.i64(i64 %a, i64 %b, i1 %m)
define i64 @srem_i64(i64 %a, i64 %b, i1 %m) {
  %r = call i64 @llvm.masked.srem.i64(i64 %a, i64 %b, i1 %m)
  ret i64 %r
}

; CHECK-LABEL: define i128 @sdiv_i128(
; CHECK: %r = call i128 @llvm.masked.sdiv.i128(i128 %a, i128 %b, i1 %m)
define i128 @sdiv_i128(i128 %a, i128 %b, i1 %m) {
  %r = call i128 @llvm.masked.sdiv.i128(i128 %a, i128 %b, i1 %m)
  ret i128 %r
}
