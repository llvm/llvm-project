; RUN: opt < %s -passes=instcombine -S | grep "store.*addrspace(1)"
; PR3335
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i32 @test(ptr %P) nounwind {
entry:
  %Q = addrspacecast ptr %P to ptr addrspace(1)
  store i32 0, ptr addrspace(1) %Q, align 4
  ret i32 0
}
