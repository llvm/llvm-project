; RUN: opt -mtriple=x86_64-unknown-linux -passes=load-store-vectorizer -S -o - %s | FileCheck %s
; RUN: opt -mtriple=x86_64-unknown-linux -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

%struct.buffer_t = type { i32, i8* }

; Check an i32 and i8* get vectorized, and that the two accesses
; (load into buff.val and store to buff.p) preserve their order.
; Vectorized loads should be inserted at the position of the first load,
; and instructions which were between the first and last load should be
; reordered preserving their relative order inasmuch as possible.

; CHECK-LABEL: @preserve_order_32(
; CHECK: load <2 x i32>
; CHECK: %buff.val = load i8
; CHECK: store i8 0
define void @preserve_order_32(%struct.buffer_t* noalias %buff) #0 {
entry:
  %tmp1 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i32 0, i32 1
  %buff.p = load i8*, i8** %tmp1
  %buff.val = load i8, i8* %buff.p
  store i8 0, i8* %buff.p, align 8
  %tmp0 = getelementptr inbounds %struct.buffer_t, %struct.buffer_t* %buff, i32 0, i32 0
  %buff.int = load i32, i32* %tmp0, align 8
  ret void
}

attributes #0 = { nounwind }
