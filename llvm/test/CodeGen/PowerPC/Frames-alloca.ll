; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {stw r31, 20(r1)}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -enable-ppc32-regscavenger | \
; RUN:   grep {stwu r1, -80(r1)}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {lwz r1, 0(r1)}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {lwz r31, 20(r1)}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {stw r31, 20(r1)}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim -enable-ppc32-regscavenger | \
; RUN:   grep {stwu r1, -80(r1)}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {lwz r1, 0(r1)}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {lwz r31, 20(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {std r31, 40(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {stdu r1, -112(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {ld r1, 0(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {ld r31, 40(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {std r31, 40(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {stdu r1, -112(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {ld r1, 0(r1)}
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {ld r31, 40(r1)}

define i32* @f1(i32 %n) {
	%tmp = alloca i32, i32 %n		; <i32*> [#uses=1]
	ret i32* %tmp
}
