;; Check that we remove the exact MCInst number from --asm-verbose output
; RUN: llc < %s -mtriple=i686-unknown-unknown --asm-show-inst | FileCheck %s --check-prefix=VERBOSE
; RUN: llc < %s -mtriple=i686-unknown-unknown | FileCheck %s --check-prefix=CHECK

define i8 @add_i8(i8 %a) nounwind {
  %add = add i8 %a, 2
  ret i8 %add
}

define i32 @add_i32(i32 %a) nounwind {
  %add = add i32 %a, 2
  ret i32 %add
}
