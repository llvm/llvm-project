; RUN: llvm-reduce --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; We can't drop arguments if they are used by terminator instructions.

define i32 @t(i32 %a0, i32 %a1, i32 %a2) {
; CHECK-ALL-LABEL: @t
; CHECK-FINAL-NOT: %a1
;
; CHECK-INTERESTINGNESS: ret i32
; CHECK-FINAL: ret i32 undef

  ret i32 %a1
}
