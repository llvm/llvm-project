## We currently parse but ignore .arch directives.
# RUN: llvm-mc -triple=x86_64 %s | FileCheck /dev/null --implicit-check-not=.arch

.arch i286
.arch generic32

.arch .avx512vl
.arch .noavx512bw
.arch .nop
.arch .sse4.2
