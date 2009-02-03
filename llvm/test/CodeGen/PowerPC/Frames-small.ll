; RUN: llvm-as < %s | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -o %t1 -f
; RUN  not grep {stw r31, 20(r1)} %t1
; RUN: grep {stwu r1, -16448(r1)} %t1
; RUN: grep {addi r1, r1, 16448} %t1
; RUN: llvm-as < %s | \
; RUN: not grep {lwz r31, 20(r1)}
; RUN: llvm-as < %s | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim \
; RUN:   -o %t2 -f
; RUN: grep {stw r31, 20(r1)} %t2
; RUN: grep {stwu r1, -16448(r1)} %t2
; RUN: grep {addi r1, r1, 16448} %t2
; RUN: grep {lwz r31, 20(r1)} %t2
; RUN: llvm-as < %s | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -o %t3 -f
; RUN: not grep {std r31, 40(r1)} %t3
; RUN: grep {stdu r1, -16496(r1)} %t3
; RUN: grep {addi r1, r1, 16496} %t3
; RUN: not grep {ld r31, 40(r1)} %t3
; RUN: llvm-as < %s | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim \
; RUN:   -o %t4 -f
; RUN: grep {std r31, 40(r1)} %t4
; RUN: grep {stdu r1, -16496(r1)} %t4
; RUN: grep {addi r1, r1, 16496} %t4
; RUN: grep {ld r31, 40(r1)} %t4

define i32* @f1() {
        %tmp = alloca i32, i32 4095             ; <i32*> [#uses=1]
        ret i32* %tmp
}

