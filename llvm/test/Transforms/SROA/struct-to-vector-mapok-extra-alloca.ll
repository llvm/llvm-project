; RUN: opt -passes='default<O3>' -S %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define void @"_ZN102_$LT$futures_util..stream..try_stream..MapOk$LT$St$C$F$GT$$u20$as$u20$futures_core..stream..Stream$GT$9poll_next17h555df33481d9c33cE"
; CHECK: [[TMP:%.*]] = alloca [11 x i64], align 8
; CHECK: call void @llvm.lifetime.start.p0(ptr nonnull [[TMP]])
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(88) [[TMP]], ptr noundef nonnull align 1 dereferenceable(88) %0, i64 88, i1 false)
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(88) %0, ptr noundef nonnull align 8 dereferenceable(88) [[TMP]], i64 88, i1 false)
; CHECK: call void @llvm.lifetime.end.p0(ptr nonnull [[TMP]])
define void @"_ZN102_$LT$futures_util..stream..try_stream..MapOk$LT$St$C$F$GT$$u20$as$u20$futures_core..stream..Stream$GT$9poll_next17h555df33481d9c33cE"(ptr %0) {
  call void @"_ZN101_$LT$futures_util..stream..stream..map..Map$LT$St$C$F$GT$$u20$as$u20$futures_core..stream..Stream$GT$9poll_next17h5aa844c062b2077eE"(ptr %0)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

define void @"_ZN101_$LT$futures_util..stream..stream..map..Map$LT$St$C$F$GT$$u20$as$u20$futures_core..stream..Stream$GT$9poll_next17h5aa844c062b2077eE"(ptr %0) {
  %.sroa.5 = alloca [11 x i64], align 8
  %2 = load i64, ptr %0, align 8
  %3 = icmp eq i64 %2, 0
  br i1 %3, label %"_ZN122_$LT$futures_util..fns..MapOkFn$LT$F$GT$$u20$as$u20$futures_util..fns..FnMut1$LT$core..result..Result$LT$T$C$E$GT$$GT$$GT$8call_mut17h252763b5559d12fbE.exit", label %4

4:                                                ; preds = %1
  call void @llvm.memcpy.p0.p0.i64(ptr %.sroa.5, ptr %0, i64 88, i1 false)
  br label %"_ZN122_$LT$futures_util..fns..MapOkFn$LT$F$GT$$u20$as$u20$futures_util..fns..FnMut1$LT$core..result..Result$LT$T$C$E$GT$$GT$$GT$8call_mut17h252763b5559d12fbE.exit"

"_ZN122_$LT$futures_util..fns..MapOkFn$LT$F$GT$$u20$as$u20$futures_util..fns..FnMut1$LT$core..result..Result$LT$T$C$E$GT$$GT$$GT$8call_mut17h252763b5559d12fbE.exit": ; preds = %4, %1
  call void @llvm.memcpy.p0.p0.i64(ptr %0, ptr %.sroa.5, i64 88, i1 false)
  ret void
}

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
