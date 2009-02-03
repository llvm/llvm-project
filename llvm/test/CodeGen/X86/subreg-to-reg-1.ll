; RUN: llvm-as < %s | llc -march=x86-64 | grep {movl	%e.\*, %e.\*} | count 1

; Don't eliminate or coalesce away the explicit zero-extension!

define i64 @foo(i64 %a) {
  %b = add i64 %a, 4294967295
  %c = and i64 %b, 4294967295
  %d = add i64 %c, 1
  ret i64 %d
}
