; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic | FileCheck %s

@g1 = private global i8 1
define i8* @get_g1() {
; CHECK: get_g1:
; CHECK: leaq .Lg1(%rip), %rax
  ret i8* @g1
}
