; RUN: opt -S -passes=asan < %s | FileCheck %s

; Check behavior with a non-0 default program address space. The
; constructor should be in addrspace(1) and the global_ctors should
; pass the verifier.

; CHECK: @a = internal addrspace(42) global [1 x i32] zeroinitializer, align 4
; CHECK: @llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @asan.module_ctor to ptr)], section "llvm.metadata"
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr addrspace(1), ptr }] [{ i32, ptr addrspace(1), ptr } { i32 1, ptr addrspace(1) @asan.module_ctor, ptr addrspacecast (ptr addrspace(1) @asan.module_ctor to ptr) }]

; CHECK: define internal void @asan.module_ctor() addrspace(1) #0 comdat {

target datalayout = "P1"

@a = internal addrspace(42) global [1 x i32] zeroinitializer, align 4

define i1 @b(i64 %c) addrspace(1) {
  %cast = inttoptr i64 %c to ptr addrspace(42)
  %cmp = icmp ugt ptr addrspace(42) %cast, getelementptr inbounds ([1 x i32], ptr addrspace(42) @a, i64 0, i64 0)
  ret i1 %cmp
}

!llvm.asan.globals = !{!0}
!0 = !{ptr addrspace(42) @a, null, !"a", i1 false, i1 false}
