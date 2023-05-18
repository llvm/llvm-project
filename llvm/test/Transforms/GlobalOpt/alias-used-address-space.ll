; RUN: opt -S -passes=globalopt < %s | FileCheck %s

target datalayout = "p:32:32:32-p1:16:16:16"

@c = hidden addrspace(1) global i8 42

@i = internal addrspace(1) global i8 42
; CHECK: @i = internal addrspace(1) global i8 42

@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @ca to ptr)], section "llvm.metadata"
; CHECK-DAG: @llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @ca to ptr)], section "llvm.metadata"

@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast(ptr addrspace(1) @ia to ptr), ptr addrspacecast (ptr addrspace(1) @i to ptr)], section "llvm.metadata"
; CHECK-DAG: @llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @i to ptr), ptr addrspacecast (ptr addrspace(1) @ia to ptr)], section "llvm.metadata"

@sameAsUsed = global [1 x ptr] [ptr addrspacecast(ptr addrspace(1) @ca to ptr)]
; CHECK-DAG: @sameAsUsed = local_unnamed_addr global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @c to ptr)]

@ia = internal alias i8, ptr addrspace(1) @i
; CHECK: @ia = internal alias i8, ptr addrspace(1) @i

@ca = internal alias i8, ptr addrspace(1) @c
; CHECK: @ca = internal alias i8, ptr addrspace(1) @c

define ptr addrspace(1) @h() {
  ret ptr addrspace(1) @ca
}
