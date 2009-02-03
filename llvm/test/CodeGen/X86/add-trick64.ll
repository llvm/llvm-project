; RUN: llvm-as < %s | llc -march=x86-64 > %t
; RUN: not grep add %t
; RUN: grep subq %t | count 2

; The immediate can be encoded in a smaller way if the
; instruction is a sub instead of an add.

define i64 @foo(i64 inreg %a) nounwind {
  %b = add i64 %a, 2147483648
  ret i64 %b
}
define i64 @bar(i64 inreg %a) nounwind {
  %b = add i64 %a, 128
  ret i64 %b
}
