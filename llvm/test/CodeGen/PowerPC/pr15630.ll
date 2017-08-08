; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O0 < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define weak_odr void @_D4core6atomic49__T11atomicStoreVE4core6atomic11MemoryOrder3ThThZ11atomicStoreFNaNbKOhhZv(i8* %val_arg, i8 zeroext %newval_arg) {
entry:
  %newval = alloca i8
  %ordering = alloca i32, align 4
  store i8 %newval_arg, i8* %newval
  %tmp = load i8, i8* %newval
  store atomic volatile i8 %tmp, i8* %val_arg seq_cst, align 1
  ret void
}

; CHECK: sync
; CHECK: stb
