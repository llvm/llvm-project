; RUN: llvm-as < %s | llc -march=ppc32

define void @main() {
        %tr4 = shl i64 1, 0             ; <i64> [#uses=0]
        ret void
}

