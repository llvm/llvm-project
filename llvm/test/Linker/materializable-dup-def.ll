; Test that linking a lazy-loaded bitcode module whose definitions are still
; materializable (empty body) against an already-linked definition does not
; report "multiply defined" — see shouldLinkFromSource in LinkModules.cpp.
;

; RUN: llvm-as %p/Inputs/materializable-dup-def-dst.ll -o %t.dst.bc
; RUN: llvm-as %p/Inputs/materializable-dup-def-src.ll -o %t.src.bc
; RUN: llvm-link %t.dst.bc %t.src.bc -o %t.linked.bc
; RUN: llvm-dis %t.linked.bc -o - | FileCheck %s

; CHECK: define i32 @mat_dup_test_1() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i32 42
; CHECK-NEXT: }

; CHECK: define void @mat_dup_test_2(i32 %a, i32 %b) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %sum = add nsw i32 %a, %b
; CHECK-NEXT:   %prod = mul nsw i32 %sum, 3
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; With eager loading both sides have full definitions; the linker correctly errors.
; RUN: not llvm-link -disable-lazy-loading %t.dst.bc %t.src.bc -o %t.eager.bc 2>&1 | FileCheck --check-prefix=EAGER %s
; EAGER: error: Linking globals named 'mat_dup_test_1': symbol multiply defined!
