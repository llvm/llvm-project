; RUN: llvm-as < %s | llvm-dis | grep 1.0

define double @test() {
        ret double 1.0   ;; This should not require hex notation
}

