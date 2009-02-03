; RUN: llvm-as < %s | llc -march=x86-64 | grep mov | count 1

; Do eliminate the zero-extension instruction and rely on
; x86-64's implicit zero-extension!

define i64 @foo(i32* %p) nounwind {
  %t = load i32* %p
  %n = add i32 %t, 1
  %z = zext i32 %n to i64
  ret i64 %z
}
