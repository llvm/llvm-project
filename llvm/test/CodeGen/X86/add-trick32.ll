; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: not grep add %t
; RUN: grep subl %t | count 1

; The immediate can be encoded in a smaller way if the
; instruction is a sub instead of an add.

define i32 @foo(i32 inreg %a) nounwind {
  %b = add i32 %a, 128
  ret i32 %b
}
