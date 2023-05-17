; RUN: llc -verify-machineinstrs < %s

; This test formerly failed due to no handling for a ppc_fp128 undef.

target datalayout = "E-p:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:64:128-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "powerpc-unknown-linux-gnu"

%core.time.TickDuration.37.125 = type { i64 }

define weak_odr fastcc i64 @_D4core4time12TickDuration30__T2toVAyaa7_7365636f6e6473TlZ2toMxFNaNbNfZl(ptr %.this_arg) {
entry:
  br i1 undef, label %noassert, label %assert

assert:                                           ; preds = %entry
  unreachable

noassert:                                         ; preds = %entry
  %tmp9 = fptosi ppc_fp128 undef to i64
  ret i64 %tmp9
}
