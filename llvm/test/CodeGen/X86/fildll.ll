; RUN: llc < %s -mtriple=i686-- -x86-asm-syntax=att -mattr=-sse2 | grep fildll | count 2

define fastcc double @sint64_to_fp(i64 %X) {
        %R = sitofp i64 %X to double            ; <double> [#uses=1]
        ret double %R
}

define fastcc double @uint64_to_fp(i64 %X) {
        %R = uitofp i64 %X to double            ; <double> [#uses=1]
        ret double %R
}

