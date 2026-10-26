; RUN: not llc -mtriple=x86_64-linux-gnu -O0 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64-linux-gnu -O2 < %s 2>&1 | FileCheck %s

; After an error in the asm of a callbr, the indirect target's outputs are
; undefined, and nothing branches there.

; CHECK: error: could not allocate output register for constraint '{foo}'
define i32 @bad_output_register(i1 %c) {
entry:
  br i1 %c, label %asm, label %cleanup

asm:
  %direct = callbr i32 asm "", "={foo},!i"()
          to label %cleanup [label %indirect]

cleanup:
  ret i32 0

indirect:
  ret i32 %direct
}

; CHECK: error: invalid operand for inline asm constraint 'i'
define i32 @bad_immediate(i1 %c, i32 %x) {
entry:
  br i1 %c, label %asm, label %cleanup

asm:
  %direct = callbr i32 asm "", "=r,i,!i"(i32 %x)
          to label %cleanup [label %indirect]

cleanup:
  ret i32 0

indirect:
  ret i32 %direct
}
