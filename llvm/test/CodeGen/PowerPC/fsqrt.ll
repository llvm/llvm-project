; fsqrt should be generated when the fsqrt feature is enabled, but not 
; otherwise.

; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=+fsqrt | \
; RUN:   grep "fsqrt f1, f1"
; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -march=ppc32 -mtriple=powerpc-apple-darwin8 -mcpu=g5 | \
; RUN:   grep "fsqrt f1, f1"
; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=-fsqrt | \
; RUN:   not grep "fsqrt f1, f1"
; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -march=ppc32 -mtriple=powerpc-apple-darwin8 -mcpu=g4 | \
; RUN:   not grep "fsqrt f1, f1"

declare double @llvm.sqrt.f64(double)

define double @X(double %Y) {
        %Z = call double @llvm.sqrt.f64( double %Y )            ; <double> [#uses=1]
        ret double %Z
}

