; It's a bad idea to go recursively traipsing through types without a safety 
; net.

; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%M = type { %M*, i32* }" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc

%T1 = type opaque
%M = type { %M*, %T1* }

