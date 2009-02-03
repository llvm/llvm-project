; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {store.*addrspace(1)}
; PR3335
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i32 @test(i32* %P) nounwind {
entry:
  %Q = bitcast i32* %P to i32 addrspace(1)*
  store i32 0, i32 addrspace(1)* %Q, align 4
  ret i32 0
}
