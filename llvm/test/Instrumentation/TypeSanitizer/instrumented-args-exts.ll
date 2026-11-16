; Test extensions of arguments to instrumented functions.
;
; RUN: opt -passes='tysan' -mtriple=s390x-unknown-linux -S %s | FileCheck %s

define void @fun() {
  ret void
}

; CHECK: declare void @__tysan_check(ptr, i32 signext, ptr, i32 signext)
; CHECK: declare void @__tysan_instrument_mem_inst(ptr, ptr, i64, i1 zeroext)
; CHECK: declare void @__tysan_instrument_with_shadow_update(ptr, ptr, i1 zeroext, i64, i32 signext)
