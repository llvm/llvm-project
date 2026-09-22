; Test extensions of arguments to instrumented function.
;
; RUN: opt -passes='dfsan' -mtriple=s390x-unknown-linux -S %s | FileCheck %s

define void @fun() {
  ret void
}

; CHECK: declare void @__dfsan_mem_shadow_origin_conditional_exchange(i8 zeroext, ptr, ptr, ptr, i64)
