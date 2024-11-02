; RUN: opt -passes=load-store-vectorizer -march=nvptx64 -mcpu=sm_35 -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: @foo
define i32 @foo(i32* %ptr) {
  %ptr1 = getelementptr i32, i32* %ptr, i32 1
  %p1 = addrspacecast i32* %ptr1 to i32 addrspace(1)*
  ; CHECK: load <2 x i32>, <2 x i32>* %{{[0-9]+}}, align 8, !invariant.load !0
  %v0 = load i32, i32* %ptr, align 8, !invariant.load !0
  %v1 = load i32, i32* %ptr1, align 4, !invariant.load !0
  %sum = add i32 %v0, %v1
  ret i32 %sum
}

!0 = !{}
