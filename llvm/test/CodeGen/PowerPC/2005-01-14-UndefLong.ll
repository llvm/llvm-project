; RUN: llc -verify-machineinstrs < %s -march=ppc32

define i64 @test() {
        ret i64 undef
}
