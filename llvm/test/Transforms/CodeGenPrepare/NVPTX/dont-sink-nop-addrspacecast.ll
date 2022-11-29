; RUN: opt -S -codegenprepare < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: @test
define i64 @test(i1 %pred, ptr %ptr) {
; CHECK: addrspacecast
  %ptr_as1 = addrspacecast ptr %ptr to ptr addrspace(1)
  br i1 %pred, label %l1, label %l2
l1:
; CHECK-LABEL: l1:
; CHECK-NOT: addrspacecast
  %v1 = load i64, ptr %ptr
  ret i64 %v1
l2:
  ; CHECK-LABEL: l2:
  ; CHECK-NOT: addrspacecast
  %v2 = load i64, ptr addrspace(1) %ptr_as1
  ret i64 %v2
}
