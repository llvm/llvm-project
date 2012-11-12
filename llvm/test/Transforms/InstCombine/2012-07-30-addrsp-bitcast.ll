; RUN: opt < %s -instcombine -S | FileCheck %s
; CHECK: bitcast

@base = internal addrspace(3) unnamed_addr global [16 x i32] zeroinitializer, align 16
declare void @foo(i32*)

define void @test() nounwind {
  call void @foo(i32* getelementptr (i32* bitcast ([16 x i32] addrspace(3)* @base to i32*), i64 2147483647)) nounwind
  ret void
}
