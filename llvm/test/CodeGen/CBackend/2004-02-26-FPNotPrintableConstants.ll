; This is a non-normal FP value
; RUN: llvm-as < %s | llc -march=c | grep FPConstant | grep static

define float @func() {
        ret float 0xFFF0000000000000
}

define double @func2() {
        ret double 0xFF20000000000000
}

