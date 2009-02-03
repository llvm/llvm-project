; RUN: llvm-as < %s | llc -march=c | grep {"m"(llvm_cbe_newcw))}
; PR2407

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define void @foo() {
  %newcw = alloca i16             ; <i16*> [#uses=2]
  call void asm sideeffect "fldcw $0", "*m,~{dirflag},~{fpsr},~{flags}"( i16*
%newcw ) nounwind 
  ret void
}
