; RUN: opt < %s -passes='function(csan)' -S -mtriple=nvptx64-nvidia-cuda | FileCheck %s

define void @shared_alloca() sanitize_concurrency {
entry:
  %p = alloca i32, align 4, addrspace(3)
  store i32 0, ptr addrspace(3) %p, align 4
  ret void
}
; CHECK-LABEL: @shared_alloca(
; CHECK: call void @__csan_write4(ptr %{{.*}}, i32 0)
; CHECK: store i32 0, ptr addrspace(3) %p, align 4

define void @local_alloca() sanitize_concurrency {
entry:
  %p = alloca i32, align 4, addrspace(5)
  store i32 0, ptr addrspace(5) %p, align 4
  ret void
}
; CHECK-LABEL: @local_alloca(
; CHECK-NEXT: entry:
; CHECK-NEXT: %p = alloca i32, align 4, addrspace(5)
; CHECK-NEXT: store i32 0, ptr addrspace(5) %p, align 4
; CHECK-NEXT: ret void
