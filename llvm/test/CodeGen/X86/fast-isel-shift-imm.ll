; RUN: llvm-as < %s | llc -march=x86 -fast | grep {sarl	\$80, %eax}
; PR3242

define i32 @foo(i32 %x) nounwind {
  %y = ashr i32 %x, 50000
  ret i32 %y
}
