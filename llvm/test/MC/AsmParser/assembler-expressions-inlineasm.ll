; RUN: not llc -mtriple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=x86_64 -no-integrated-as < %s | FileCheck %s --check-prefix=GAS
; RUN: llc -mtriple=x86_64 -filetype=obj %s -o - | llvm-objdump -d - |  FileCheck %s --check-prefix=DISASM

; GAS: nop;  .if . - foo==1; nop;.endif

; CHECK: <inline asm>:1:17: error: expected absolute expression

; DISASM:      <main>:
; DISASM-NEXT:   nop
; DISASM-NEXT:   nop
; DISASM-NEXT:   xorl %eax, %eax
; DISASM-NEXT:   retq

define i32 @main() local_unnamed_addr {
  tail call void asm sideeffect "foo: nop;  .if . - foo==1;  nop;.endif", "~{dirflag},~{fpsr},~{flags}"()
  ret i32 0
}
