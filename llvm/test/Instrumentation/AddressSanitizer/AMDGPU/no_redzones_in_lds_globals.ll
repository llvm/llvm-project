; RUN: opt < %s -passes=asan -S | FileCheck %s
target triple = "amdgpu7.00-amd-amdhsa"

@G10 = addrspace(3) global [10 x i8] zeroinitializer, align 1
; CHECK-NOT: @G10 = addrspace(3) global { [10 x i8], [* x i8] }
