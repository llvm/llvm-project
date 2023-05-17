; RUN: llc < %s -mtriple=arm64-apple-darwin  | FileCheck %s
; Checks for conditional branch b.vs

; Function Attrs: nounwind
define i32 @add(i32, i32) {
entry:
  %2 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue { i32, i1 } %2, 1
  br i1 %3, label %6, label %4

; <label>:4                                       ; preds = %entry
  %5 = extractvalue { i32, i1 } %2, 0
  ret i32 %5

; <label>:6                                       ; preds = %entry
  tail call void @llvm.trap()
  unreachable
; CHECK: b.vs
}

%S64 = type <{ i64 }>
%S32 = type <{ i32 }>
%Sstruct = type <{ %S64, %S32 }>

; Checks for compfail when optimizing csincr-cbz sequence

define { i64, i1 } @foo(ptr , ptr , i1, i64) {
entry:
  %.sroa.0 = alloca i72, align 16
  %4 = load i64, ptr %1, align 8
  %.repeatedValue.value = getelementptr inbounds %Sstruct, ptr %1, i64 0, i32 1, i32 0
  %5 = load i32, ptr %.repeatedValue.value, align 8
  %6 = icmp eq i64 %4, 0
  br label %7

; <label>:7                                      ; preds = %entry
  %.mask58 = and i32 %5, -2048
  %8 = icmp eq i32 %.mask58, 55296
  %.not134 = xor i1 %8, true
  %9 = icmp eq i32 %5, 1114112
  %or.cond135 = and i1 %9, %.not134
  br i1 %or.cond135, label %10, label %.loopexit

; <label>:10                                      ; preds = %7
  %11 = and i32 %5, -2048
  %12 = icmp eq i32 %11, 55296
  br i1 %12, label %.loopexit, label %10


.loopexit:                                        ; preds = %.entry,%7,%10
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

; Function Attrs: noreturn nounwind
declare void @llvm.trap()
