; RUN: opt < %s -dfsan -S | FileCheck %s --check-prefixes=CHECK,CHECK_NO_ORIGIN -DSHADOW_XOR_MASK=87960930222080 --dump-input-context=100
; RUN: opt < %s -dfsan -dfsan-track-origins=1  -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN -DSHADOW_XOR_MASK=87960930222080 --dump-input-context=100
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [100 x i64]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [100 x i64]
; CHECK: @__dfsan_arg_origin_tls = external thread_local(initialexec) global [200 x i32]
; CHECK: @__dfsan_retval_origin_tls = external thread_local(initialexec) global i32
; CHECK_NO_ORIGIN: @__dfsan_track_origins = weak_odr constant i32 0
; CHECK_ORIGIN: @__dfsan_track_origins = weak_odr constant i32 1
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i8 @load(i8* %p) {
  ; CHECK-LABEL: define i8 @load.dfsan
  ; CHECK: xor i64 {{.*}}, [[SHADOW_XOR_MASK]]
  ; CHECK: ret i8 %a
  %a = load i8, i8* %p
  ret i8 %a
}

define void @store(i8* %p) {
  ; CHECK-LABEL: define void @store.dfsan
  ; CHECK: xor i64 {{.*}}, [[SHADOW_XOR_MASK]]
  ; CHECK: ret void
  store i8 0, i8* %p
  ret void
}

; CHECK: declare void @__dfsan_load_callback(i[[#SBITS]], i8*)
; CHECK: declare void @__dfsan_store_callback(i[[#SBITS]], i8*)
; CHECK: declare void @__dfsan_mem_transfer_callback(i[[#SBITS]]*, i64)
; CHECK: declare void @__dfsan_cmp_callback(i[[#SBITS]])

; CHECK: ; Function Attrs: nounwind readonly
; CHECK-NEXT: declare zeroext i[[#SBITS]] @__dfsan_union_load(i[[#SBITS]]*, i64)

; CHECK: ; Function Attrs: nounwind readonly
; CHECK-NEXT: declare zeroext i64 @__dfsan_load_label_and_origin(i8*, i64)

; CHECK: declare void @__dfsan_unimplemented(i8*)
; CHECK: declare void @__dfsan_set_label(i[[#SBITS]] zeroext, i32 zeroext, i8*, i64)
; CHECK: declare void @__dfsan_nonzero_label()
; CHECK: declare void @__dfsan_vararg_wrapper(i8*)
; CHECK: declare zeroext i32 @__dfsan_chain_origin(i32 zeroext)
; CHECK: declare zeroext i32 @__dfsan_chain_origin_if_tainted(i[[#SBITS]] zeroext, i32 zeroext)
; CHECK: declare void @__dfsan_mem_origin_transfer(i8*, i8*, i64)
; CHECK: declare void @__dfsan_maybe_store_origin(i[[#SBITS]] zeroext, i8*, i64, i32 zeroext)
