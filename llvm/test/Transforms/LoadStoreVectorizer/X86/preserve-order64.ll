; RUN: opt -mtriple=x86-linux -load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

%struct.buffer_t = type { i64, i8* }

; Check an i64 and i8* get vectorized, and that
; the two accesses (load into buff.val and store to buff.p) preserve their order.

; CHECK-LABEL: @preserve_order_64(
; CHECK: load <2 x i64>
; CHECK: %buff.val = load i8
; CHECK: store i8 0
define void @preserve_order_64(%struct.buffer_t* noalias %buff) #0 {
entry:
  %tmp1 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i64 0, i32 1
  %buff.p = load i8*, i8** %tmp1, align 8
  %buff.val = load i8, i8* %buff.p, align 8
  store i8 0, i8* %buff.p, align 8
  %tmp0 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i64 0, i32 0
  %buff.int = load i64, i64* %tmp0, align 8
  ret void
}

attributes #0 = { nounwind }
