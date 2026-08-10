; RUN: opt -data-layout=A5 -passes=globalopt %s -S -o - | FileCheck %s

; CHECK-NOT: @g
@g = internal addrspace(1) global ptr zeroinitializer

; CHECK: @llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
   [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]

; CHECK-NOT: @ctor
define internal void @ctor()  {
  %addr = alloca i32, align 8, addrspace(5)
  %tmp = addrspacecast ptr addrspace(5) %addr to ptr
  store ptr %tmp, ptr addrspace(1) @g
  ret void
}

