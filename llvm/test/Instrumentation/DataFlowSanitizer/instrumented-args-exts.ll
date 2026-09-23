; Test extensions of arguments to instrumented functions.
;
; RUN: opt -passes='dfsan' -mtriple=x86_64-unknown-linux-gnu -S %s | FileCheck %s
; RUN: opt -passes='dfsan' -mtriple=s390x-unknown-linux -S %s | FileCheck %s --check-prefix=SYSTEMZ

define void @fun() {
  ret void
}


; CHECK: declare void @__dfsan_load_callback(i8, ptr)
; CHECK: declare void @__dfsan_store_callback(i8, ptr)
; CHECK: declare void @__dfsan_cmp_callback(i8)
; CHECK: declare void @__dfsan_conditional_callback(i8)
; CHECK: declare void @__dfsan_conditional_callback_origin(i8, i32)
; CHECK: declare void @__dfsan_reaches_function_callback(i8, ptr, i32, ptr)
; CHECK: declare void @__dfsan_reaches_function_callback_origin(i8, i32, ptr, i32, ptr)
; CHECK: declare void @__dfsan_set_label(i8, i32, ptr, i64)
; CHECK: declare zeroext i32 @__dfsan_chain_origin(i32)
; CHECK: declare zeroext i32 @__dfsan_chain_origin_if_tainted(i8, i32)
; CHECK: declare void @__dfsan_mem_shadow_origin_conditional_exchange(i8, ptr, ptr, ptr, i64)
; CHECK: declare void @__dfsan_maybe_store_origin(i8, ptr, i64, i32)

; SYSTEMZ: declare void @__dfsan_load_callback(i8 zeroext, ptr)
; SYSTEMZ: declare void @__dfsan_store_callback(i8 zeroext, ptr)
; SYSTEMZ: declare void @__dfsan_cmp_callback(i8 zeroext)
; SYSTEMZ: declare void @__dfsan_conditional_callback(i8 zeroext)
; SYSTEMZ: declare void @__dfsan_conditional_callback_origin(i8 zeroext, i32 zeroext)
; SYSTEMZ: declare void @__dfsan_reaches_function_callback(i8 zeroext, ptr, i32 zeroext, ptr)
; SYSTEMZ: declare void @__dfsan_reaches_function_callback_origin(i8 zeroext, i32 zeroext, ptr, i32 zeroext, ptr)
; SYSTEMZ: declare void @__dfsan_set_label(i8 zeroext, i32 zeroext, ptr, i64)
; SYSTEMZ: declare zeroext i32 @__dfsan_chain_origin(i32 zeroext)
; SYSTEMZ: declare zeroext i32 @__dfsan_chain_origin_if_tainted(i8 zeroext, i32 zeroext)
; SYSTEMZ: declare void @__dfsan_mem_shadow_origin_conditional_exchange(i8 zeroext, ptr, ptr, ptr, i64)
; SYSTEMZ: declare void @__dfsan_maybe_store_origin(i8 zeroext, ptr, i64, i32 zeroext)

