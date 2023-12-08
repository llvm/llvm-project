; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

declare void @llvm.va_start(ptr)

define internal i32 @va_func(i32 %a, i32 %b, ...) {
  %valist = alloca i8
  call void @llvm.va_start(ptr %valist)

  ret i32 %b
}

; Function derived from AArch64 ABI, where 8 integer arguments go in
; registers but the 9th goes on the stack. We really don't want to put
; just 7 args in registers and then start on the stack since any
; va_arg implementation already present in va_func won't be expecting
; it.
define i32 @call_va(i32 %in) {
  %stacked = alloca i32
  store i32 42, ptr %stacked
  %res = call i32(i32, i32, ...) @va_func(i32 %in, i32 %in, [6 x i32] poison, ptr byval(i32) %stacked)
  ret i32 %res
; CHECK: call i32 (i32, i32, ...) @va_func(i32 poison, i32 %in, [6 x i32] poison, ptr byval(i32) %stacked)
}

define internal i32 @va_deadret_func(i32 %a, i32 %b, ...) {
  %valist = alloca i8
  call void @llvm.va_start(ptr %valist)

  ret i32 %a
}

define void @call_deadret(i32 %in) {
  %stacked = alloca i32
  store i32 42, ptr %stacked
  call i32 (i32, i32, ...) @va_deadret_func(i32 poison, i32 %in, [6 x i32] poison, ptr byval(i32) %stacked)
  ret void
; CHECK: call void (i32, i32, ...) @va_deadret_func(i32 poison, i32 poison, [6 x i32] poison, ptr byval(i32) %stacked)
}
