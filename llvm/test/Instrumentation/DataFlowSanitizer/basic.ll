; RUN: opt < %s -passes=dfsan -S | FileCheck %s --check-prefixes=CHECK,CHECK_NO_ORIGIN -DSHADOW_XOR_MASK=87960930222080 --dump-input-context=100
; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1  -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN1 -DSHADOW_XOR_MASK=87960930222080 --dump-input-context=100
; RUN: opt < %s -passes=dfsan -dfsan-track-origins=2  -S | FileCheck %s --check-prefixes=CHECK_ORIGIN2 -DSHADOW_XOR_MASK=87960930222080 --dump-input-context=100
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [100 x i64]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [100 x i64]
; CHECK: @__dfsan_arg_origin_tls = external thread_local(initialexec) global [200 x i32]
; CHECK: @__dfsan_retval_origin_tls = external thread_local(initialexec) global i32
; CHECK_NO_ORIGIN: @__dfsan_track_origins = weak_odr constant i32 0
; CHECK_ORIGIN1: @__dfsan_track_origins = weak_odr constant i32 1
; CHECK_ORIGIN2: @__dfsan_track_origins = weak_odr constant i32 2
define i8 @load(ptr %p) {
  ; CHECK-LABEL: define i8 @load.dfsan
  ; CHECK: xor i64 {{.*}}, [[SHADOW_XOR_MASK]]
  ; CHECK: ret i8 %a
  %a = load i8, ptr %p
  ret i8 %a
}

define void @store(ptr %p) {
  ; CHECK-LABEL: define void @store.dfsan
  ; CHECK: xor i64 {{.*}}, [[SHADOW_XOR_MASK]]
  ; CHECK: ret void
  store i8 0, ptr %p
  ret void
}

; CHECK: declare void @__dfsan_load_callback(i8 zeroext, ptr)
; CHECK: declare void @__dfsan_store_callback(i8 zeroext, ptr)
; CHECK: declare void @__dfsan_mem_transfer_callback(ptr, i64)
; CHECK: declare void @__dfsan_cmp_callback(i8 zeroext)

; CHECK: ; Function Attrs: nounwind memory(read)
; CHECK-NEXT: declare zeroext i8 @__dfsan_union_load(ptr, i64)

; CHECK: ; Function Attrs: nounwind memory(read)
; CHECK-NEXT: declare zeroext i64 @__dfsan_load_label_and_origin(ptr, i64)

; CHECK: declare void @__dfsan_unimplemented(ptr)
; CHECK: declare void @__dfsan_set_label(i8 zeroext, i32 zeroext, ptr, i64)
; CHECK: declare void @__dfsan_nonzero_label()
; CHECK: declare void @__dfsan_vararg_wrapper(ptr)
; CHECK: declare zeroext i32 @__dfsan_chain_origin(i32 zeroext)
; CHECK: declare zeroext i32 @__dfsan_chain_origin_if_tainted(i8 zeroext, i32 zeroext)
; CHECK: declare void @__dfsan_mem_origin_transfer(ptr, ptr, i64)
; CHECK: declare void @__dfsan_maybe_store_origin(i8 zeroext, ptr, i64, i32 zeroext)
