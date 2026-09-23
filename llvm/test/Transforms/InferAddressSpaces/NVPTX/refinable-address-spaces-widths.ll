; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes=infer-address-spaces %s | FileCheck %s

; Pointer conversions must remain addrspacecasts, including mixed-width casts.
target datalayout = "e-p:64:64-p3:32:32-p7:32:32"

define i32 @nested_local(ptr addrspace(3) %base) {
; CHECK-LABEL: @nested_local(
; CHECK-NEXT: %p = getelementptr inbounds i32, ptr addrspace(3) %base, i64 1
; CHECK-NEXT: %value = load i32, ptr addrspace(3) %p, align 4
; CHECK-NEXT: ret i32 %value
  %generic = addrspacecast ptr addrspace(3) %base to ptr
  %cluster = addrspacecast ptr %generic to ptr addrspace(7)
  %p = getelementptr inbounds i32, ptr addrspace(7) %cluster, i64 1
  %value = load i32, ptr addrspace(7) %p, align 4
  ret i32 %value
}

; Integer round trips cannot stand in for a non-noop address space conversion.
define i32 @integer_roundtrip(ptr addrspace(3) %base) {
; CHECK-LABEL: @integer_roundtrip(
; CHECK: %bits = ptrtoint ptr addrspace(3) %base to i32
; CHECK-NEXT: %p = inttoptr i32 %bits to ptr addrspace(7)
; CHECK-NEXT: %value = load i32, ptr addrspace(7) %p, align 4
  %bits = ptrtoint ptr addrspace(3) %base to i32
  %p = inttoptr i32 %bits to ptr addrspace(7)
  %value = load i32, ptr addrspace(7) %p, align 4
  ret i32 %value
}
