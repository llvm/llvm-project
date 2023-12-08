; RUN: llc -O0 -mcpu=i386 -mattr=-sse,-mmx < %s
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.5.1 20100510 (prerelease) LLVM: 104604:104605\22"

define i32 @f2(double %x) nounwind {
entry:
  %0 = load double, ptr undef, align 64               ; <double> [#uses=1]
  %1 = fptoui double %0 to i16                    ; <i16> [#uses=1]
  %2 = zext i16 %1 to i32                         ; <i32> [#uses=1]
  %3 = add nsw i32 0, %2                          ; <i32> [#uses=1]
  store i32 %3, ptr undef, align 1
  ret i32 0
}
