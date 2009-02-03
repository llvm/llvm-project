; RUN: llvm-as < %s | llc -march=ppc32

define i64 @test() {
        ret i64 undef
}
