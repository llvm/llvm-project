; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   grep align.4 | count 1
; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   grep align.2 | count 1
; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   grep align.3 | count 1

@A = global <4 x i32> < i32 10, i32 20, i32 30, i32 40 >                ; <<4 x i32>*> [#uses=0]
@B = global float 1.000000e+02          ; <float*> [#uses=0]
@C = global double 2.000000e+03         ; <double*> [#uses=0]

