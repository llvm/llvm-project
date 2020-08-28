; This testcase checks to make sure that the assembler can handle floating 
; point constants in IEEE hex format. This also checks that the disassembler,
; when presented with a FP constant that cannot be represented exactly in 
; exponential form, outputs it correctly in hex format.  This is a distillation
; of the bug that was causing the Olden Health benchmark to output incorrect
; results!
;
; RUN: opt -instsimplify -S > %t.1 < %s
; RUN: llvm-as < %s | llvm-dis | llvm-as | opt -instsimplify | \
; RUN: llvm-dis > %t.2
; RUN: diff %t.1 %t.2
; RUN: verify-uselistorder %s

define double @test() {
        %tmp = fmul double 7.200000e+101, 0x427F4000             ; <double> [#uses=1]
        ret double %tmp
}
