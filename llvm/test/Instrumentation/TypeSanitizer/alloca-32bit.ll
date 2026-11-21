; Test TySan outlined instrumentation on a 32-bit target.
; The size argument to __tysan_instrument_mem_inst must be zero-extended
; from i32 (IntptrTy) to i64 (the runtime function's uint64_t parameter).
;
; RUN: opt -passes='tysan' -S %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"

declare void @use(ptr)

; Fixed-size alloca: size is a constant i32 that must be zext'd to i64.
define void @alloca_fixed() sanitize_type {
; CHECK-LABEL: @alloca_fixed(
; CHECK:         [[X:%.*]] = alloca [10 x i8], align 1
; CHECK-NEXT:    call void @__tysan_instrument_mem_inst(ptr [[X]], ptr null, i64 10, i1 false)
; CHECK-NEXT:    call void @use(ptr [[X]])
; CHECK-NEXT:    ret void
;
entry:
  %x = alloca [10 x i8], align 1
  call void @use(ptr %x)
  ret void
}

; Dynamic alloca: size is computed as i32 and must be zext'd to i64.
define void @alloca_dynamic(i32 %n) sanitize_type {
; CHECK-LABEL: @alloca_dynamic(
; CHECK:         [[X:%.*]] = alloca i32, i32 [[N:%.*]], align 1
; CHECK-NEXT:    [[SZ:%.*]] = mul i32 [[N]], 4
; CHECK-NEXT:    [[SZ64:%.*]] = zext i32 [[SZ]] to i64
; CHECK-NEXT:    call void @__tysan_instrument_mem_inst(ptr [[X]], ptr null, i64 [[SZ64]], i1 false)
; CHECK-NEXT:    call void @use(ptr [[X]])
; CHECK-NEXT:    ret void
;
entry:
  %x = alloca i32, i32 %n, align 1
  call void @use(ptr %x)
  ret void
}
