; RUN: opt < %s -passes=instcombine -S | FileCheck %s
; CHECK: addrspacecast

@base = internal unnamed_addr addrspace(3) global [16 x i32] zeroinitializer, align 16
declare void @foo(i32*)

define void @test() nounwind {
  call void @foo(i32* getelementptr (i32, i32* addrspacecast ([16 x i32] addrspace(3)* @base to i32*), i64 2147483647)) nounwind
  ret void
}
