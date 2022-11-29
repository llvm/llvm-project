; RUN: opt -S -codegenprepare -mtriple=amdgcn--amdhsa < %s | FileCheck %s

; CHECK-LABEL: @no_sink_local_to_flat(
; CHECK: addrspacecast
; CHECK: br
; CHECK-NOT: addrspacecast
define i64 @no_sink_local_to_flat(i1 %pred, ptr addrspace(3) %ptr) {
  %ptr_cast = addrspacecast ptr addrspace(3) %ptr to ptr
  br i1 %pred, label %l1, label %l2

l1:
  %v1 = load i64, ptr addrspace(3) %ptr
  ret i64 %v1

l2:
  %v2 = load i64, ptr %ptr_cast
  ret i64 %v2
}

; CHECK-LABEL: @no_sink_private_to_flat(
; CHECK: addrspacecast
; CHECK: br
; CHECK-NOT: addrspacecast
define i64 @no_sink_private_to_flat(i1 %pred, ptr addrspace(5) %ptr) {
  %ptr_cast = addrspacecast ptr addrspace(5) %ptr to ptr
  br i1 %pred, label %l1, label %l2

l1:
  %v1 = load i64, ptr addrspace(5) %ptr
  ret i64 %v1

l2:
  %v2 = load i64, ptr %ptr_cast
  ret i64 %v2
}


; CHECK-LABEL: @sink_global_to_flat(
; CHECK-NOT: addrspacecast
; CHECK: br
; CHECK: addrspacecast
define i64 @sink_global_to_flat(i1 %pred, ptr addrspace(1) %ptr) {
  %ptr_cast = addrspacecast ptr addrspace(1) %ptr to ptr
  br i1 %pred, label %l1, label %l2

l1:
  %v1 = load i64, ptr addrspace(1) %ptr
  ret i64 %v1

l2:
  %v2 = load i64, ptr %ptr_cast
  ret i64 %v2
}

; CHECK-LABEL: @sink_flat_to_global(
; CHECK-NOT: addrspacecast
; CHECK: br
; CHECK: addrspacecast
define i64 @sink_flat_to_global(i1 %pred, ptr %ptr) {
  %ptr_cast = addrspacecast ptr %ptr to ptr addrspace(1)
  br i1 %pred, label %l1, label %l2

l1:
  %v1 = load i64, ptr %ptr
  ret i64 %v1

l2:
  %v2 = load i64, ptr addrspace(1) %ptr_cast
  ret i64 %v2
}

; CHECK-LABEL: @sink_flat_to_constant(
; CHECK-NOT: addrspacecast
; CHECK: br
; CHECK: addrspacecast
define i64 @sink_flat_to_constant(i1 %pred, ptr %ptr) {
  %ptr_cast = addrspacecast ptr %ptr to ptr addrspace(4)
  br i1 %pred, label %l1, label %l2

l1:
  %v1 = load i64, ptr %ptr
  ret i64 %v1

l2:
  %v2 = load i64, ptr addrspace(4) %ptr_cast
  ret i64 %v2
}

; CHECK-LABEL: @sink_flat_to_local(
; CHECK-NOT: addrspacecast
; CHECK: br
; CHECK: addrspacecast
define i64 @sink_flat_to_local(i1 %pred, ptr %ptr) {
  %ptr_cast = addrspacecast ptr %ptr to ptr addrspace(3)
  br i1 %pred, label %l1, label %l2

l1:
  %v1 = load i64, ptr %ptr
  ret i64 %v1

l2:
  %v2 = load i64, ptr addrspace(3) %ptr_cast
  ret i64 %v2
}

; CHECK-LABEL: @sink_flat_to_private(
; CHECK-NOT: addrspacecast
; CHECK: br
; CHECK: addrspacecast
define i64 @sink_flat_to_private(i1 %pred, ptr %ptr) {
  %ptr_cast = addrspacecast ptr %ptr to ptr addrspace(5)
  br i1 %pred, label %l1, label %l2

l1:
  %v1 = load i64, ptr %ptr
  ret i64 %v1

l2:
  %v2 = load i64, ptr addrspace(5) %ptr_cast
  ret i64 %v2
}
