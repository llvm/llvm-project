; RUN: llvm-as < %s | llc -march=x86-64 -o - | not grep shr
define i128 @sl(i128 %x) {
        %t = shl i128 %x, 1
        ret i128 %t
}
