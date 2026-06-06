; RUN: opt < %s -passes=asan -S | FileCheck %s
target triple = "amdgcn-amd-amdhsa"

@G10 = addrspace(5) global [10 x i8] zeroinitializer, align 1
; CHECK-NOT: @G10 = addrspace(5) global { [10 x i8], [* x i8] }
