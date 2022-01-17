; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

define i32 @bar(i32 %A) {
  call void (i32, ...) @thunk(i32 %A, i64 47, double 1.000000e+00)
  %a = call i32 (i32, ...) @has_vastart(i32 %A, i64 47, double 1.000000e+00)
  %b = call i32 (i32, ...) @no_vastart( i32 %A, i32 %A, i32 %A, i32 %A, i64 47, double 1.000000e+00 )
  %c = add i32 %a, %b
  ret i32 %c
}
; CHECK-LABEL: define i32 @bar
; CHECK: call void (i32, ...) @thunk(i32 %A, i64 47, double 1.000000e+00)
; CHECK: call i32 (i32, ...) @has_vastart(i32 %A, i64 47, double 1.000000e+00)
; CHECK: call i32 @no_vastart(i32 %A)

declare void @thunk_target(i32 %X, ...)

define internal void @thunk(i32 %X, ...) {
  musttail call void(i32, ...) @thunk_target(i32 %X, ...)
  ret void
}
; CHECK-LABEL: define internal void @thunk(i32 %X, ...)
; CHECK: musttail call void (i32, ...) @thunk_target(i32 %X, ...)

define internal i32 @has_vastart(i32 %X, ...) {
  %valist = alloca i8
  call void @llvm.va_start(i8* %valist)
  ret i32 %X
}
; CHECK-LABEL: define internal i32 @has_vastart(i32 %X, ...)

declare void @llvm.va_start(i8*)

define internal i32 @no_vastart(i32 %X, ...) {
  ret i32 %X
}
; CHECK-LABEL: define internal i32 @no_vastart(i32 %X)
