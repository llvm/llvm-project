; RUN: llvm-as < %s | llc -march=x86 | not grep and

; These tests differ from the ones in zext-inreg-0.ll in that
; on x86-64 they do require and instructions.

; These should use movzbl instead of 'and 255'.
; This related to not having ZERO_EXTEND_REG node.

define i64 @l(i64 %d) nounwind  {
        %e = add i64 %d, 1
        %retval = and i64 %e, 1099511627775
        ret i64 %retval
}
define i64 @m(i64 %d) nounwind  {
        %e = add i64 %d, 1
        %retval = and i64 %e, 281474976710655
        ret i64 %retval
}
