; RUN: opt -passes='deadargelim,attributor' -S < %s | FileCheck %s

; The device LTO pipeline runs openmp-opt, then deadargelim, then openmp-opt
; again. By the time deadargelim runs, the first openmp-opt can have folded
; the size every remaining caller passes into the internalized
; __kmpc_alloc_shared, leaving its size argument unused. deadargelim must keep
; that argument, and keep passing it, because allocsize names it and
; heap-to-stack reads the size from the call. Dropping either one leaves an
; allocator that heap-to-stack accepts but cannot size.

target triple = "amdgcn-amd-amdhsa"

define internal noalias ptr @__kmpc_alloc_shared(i64 %Bytes) #0 {
  %p = call ptr @runtime_alloc(i64 8)
  ret ptr %p
}

; CHECK-LABEL: define void @globalized_local()
; CHECK-NEXT: [[A:%.*]] = alloca i8, i64 8, align 8, addrspace(5)
; CHECK-NOT: __kmpc_alloc_shared
; CHECK-NOT: __kmpc_free_shared
; CHECK: ret void
define void @globalized_local() {
  %p = call align 8 ptr @__kmpc_alloc_shared(i64 8)
  call void @use(ptr nocapture nofree %p) nosync nounwind willreturn
  call void @__kmpc_free_shared(ptr %p, i64 8)
  ret void
}

declare ptr @runtime_alloc(i64)
declare void @use(ptr)
declare void @__kmpc_free_shared(ptr allocptr nocapture, i64) #1

attributes #0 = { allockind("alloc,uninitialized") allocsize(0) "alloc-family"="__kmpc_alloc_shared" }
attributes #1 = { allockind("free") "alloc-family"="__kmpc_alloc_shared" }
