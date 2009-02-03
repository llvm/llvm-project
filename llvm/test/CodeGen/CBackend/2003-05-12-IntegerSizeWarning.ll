; RUN: llvm-as < %s | llc -march=c

; Apparently this constant was unsigned in ISO C 90, but not in C 99.

define i32 @foo() {
        ret i32 -2147483648
}

