; RUN: opt -S -Oz %s | FileCheck %s

@a = global i32 4

define i32 @inner() {
  %a1 = load volatile i32, i32* @a

  ; Here are enough instructions to prevent inlining, but because they are used
  ; only by the @llvm.assume intrinsic, they're free (and, thus, inlining will
  ; still happen).
  %a2 = mul i32 %a1, %a1
  %a3 = sub i32 %a1, 5
  %a4 = udiv i32 %a3, -13
  %a5 = mul i32 %a4, %a4
  %a6 = add i32 %a5, %a5
  %ca = icmp sgt i32 %a6, -7
  tail call void @llvm.assume(i1 %ca)

  ret i32 %a1
}

; @inner() should be inlined for -Oz.
; CHECK-NOT: call i1 @inner
define i32 @outer() optsize {
   %r = call i32 @inner()
   ret i32 %r
}

declare void @llvm.assume(i1) nounwind

