; RUN: opt -S -passes=globalopt < %s | FileCheck %s

target datalayout = "p:32:32:32-p1:16:16:16"

@c = addrspace(1) global i8 42

@i = internal addrspace(1) global i8 42

; CHECK: @ia = internal addrspace(1) global i8 42
@ia = internal alias i8, i8 addrspace(1)* @i

@llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(1)* @ca to i8*)], section "llvm.metadata"
; CHECK-DAG: @llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(1)* @ca to i8*)], section "llvm.metadata"

@llvm.compiler.used = appending global [2 x i8*] [i8* addrspacecast(i8 addrspace(1)* @ia to i8*), i8* addrspacecast (i8 addrspace(1)* @i to i8*)], section "llvm.metadata"
; CHECK-DAG: @llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(1)* @ia to i8*)], section "llvm.metadata"

@sameAsUsed = global [1 x i8*] [i8* addrspacecast(i8 addrspace(1)* @ca to i8*)]
; CHECK-DAG: @sameAsUsed = local_unnamed_addr global [1 x i8*] [i8* addrspacecast (i8 addrspace(1)* @c to i8*)]

@ca = internal alias i8, i8 addrspace(1)* @c
; CHECK: @ca = internal alias i8, i8 addrspace(1)* @c

define i8 addrspace(1)* @h() {
  ret i8 addrspace(1)* @ca
}
