; fsqrt should be generated when the fsqrt feature is enabled, but not 
; otherwise.

; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -mtriple=powerpc-unknown-linux-gnu -mattr=+fsqrt | FileCheck %s -check-prefix=SQRT
; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -mtriple=powerpc-unknown-linux-gnu -mattr=-fsqrt | FileCheck %s -check-prefix=NSQRT

; SQRT: X:
; SQRT: fsqrt 1, 1
; SQRT: blr

; NSQRT: X:
; NSQRT-NOT: fsqrt 1, 1
; NSQRT: blr

declare double @llvm.sqrt.f64(double)

define double @X(double %Y) {
        %Z = call double @llvm.sqrt.f64( double %Y )            ; <double> [#uses=1]
        ret double %Z
}

