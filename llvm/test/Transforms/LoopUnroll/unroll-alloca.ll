; RUN: opt -unroll-allow-partial -unroll-runtime -passes="loop-unroll<O3>" -S < %s | FileCheck %s

; Various scenarios where a loop has an alloca with a live-out lifetime use.
; LCSSA can't help here because a lifetime marker can't have a phi
; definition.
; Remove the lifetime marker(s) before unrolling, to prevent multiple defs
; and a single use.

; CHECK-LABEL: @peelit
; CHECK: alloca i32
; CHECK-NOT: call{{.*}}@llvm.lifetime

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
declare void @llvm.lifetime.end.p0(ptr captures(none)) #0

define fastcc i32 @peelit() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i64 [ 1, %bb ], [ 0, %bb1 ]
  %alloca = alloca i32, align 4
  br i1 false, label %bb2, label %bb1

bb2:                                              ; preds = %bb1
  call void @llvm.lifetime.start.p0(ptr %alloca)
  unreachable
}

; CHECK-LABEL: @partial
; CHECK: call{{.*}}umax
; CHECK: alloca i32
; CHECK-NOT: call{{.*}}@llvm.lifetime

define fastcc i32 @partial(i32 %max) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phic = phi i32 [ 0, %bb ], [ %next, %bb1 ]
  %alloca = alloca i32, align 8
  call void @llvm.lifetime.start.p0(ptr %alloca)
  store i32 %phic, ptr %alloca, align 8
  %next = add i32 %phic, 1
  %cmp = icmp ult i32 %next, %max
  br i1 %cmp, label %bb1, label %bb2

bb2:                                              ; preds = %bb1
  call void @llvm.lifetime.end.p0(ptr %alloca)
  ret i32 %phic
}

; full unroll is OK because there's no merge point.

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
