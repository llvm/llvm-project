; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -march=ppc32 | grep fmul | count 2
; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -march=ppc32 -enable-unsafe-fp-math | \
; RUN:   grep fmul | count 1

define double @foo(double %X) nounwind {
        %tmp1 = fmul double %X, 1.23
        %tmp2 = fmul double %tmp1, 4.124
        ret double %tmp2
}

