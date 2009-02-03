; RUN: llvm-as < %s | llc -march=c | \
; RUN:          grep {struct __attribute__ ((packed, aligned(} | count 4

define void @test(i32* %P) {
        %X = load i32* %P, align 1
        store i32 %X, i32* %P, align 1
        ret void
}

define void @test2(i32* %P) {
        %X = volatile load i32* %P, align 2
        volatile store i32 %X, i32* %P, align 2
        ret void
}

