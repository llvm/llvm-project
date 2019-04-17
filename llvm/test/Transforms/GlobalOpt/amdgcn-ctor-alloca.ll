; RUN: opt -data-layout=A5 -globalopt %s -S -o - | FileCheck %s

; CHECK-NOT: @g
@g = internal addrspace(1) global i32* zeroinitializer

; CHECK: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }]
   [{ i32, void ()*, i8* } { i32 65535, void ()* @ctor, i8* null }]

; CHECK-NOT: @ctor
define internal void @ctor()  {
  %addr = alloca i32, align 8, addrspace(5)
  %tmp = addrspacecast i32 addrspace(5)* %addr to i32*
  store i32* %tmp, i32* addrspace(1)* @g
  ret void
}

