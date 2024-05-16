; RUN: not llc -mtriple x86_64-unknown-linux-gnu -o %t.s -filetype=asm %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple x86_64-unknown-linux-gnu -o %t.o -filetype=obj %s 2>&1 | FileCheck %s

; Assembler-aware expression evaluation should be disabled in inline
; assembly to prevent differences in behavior between object and
; assembly output.


; CHECK: <inline asm>:1:17: error: expected absolute expression

define i32 @main() local_unnamed_addr {
  tail call void asm sideeffect "foo: nop;  .if . - foo==1;  nop;.endif", "~{dirflag},~{fpsr},~{flags}"()
  ret i32 0
}
