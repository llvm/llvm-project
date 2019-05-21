; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o - \
; RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELAX %s
; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=-relax %s -o - \
; RUN:     | llvm-readobj -r | FileCheck -check-prefix=NORELAX %s

; This test checks that a diff inserted via inline assembly only causes
; relocations when relaxation is enabled. This isn't an assembly test
; as the assembler takes a different path through LLVM, which is
; already covered by the fixups-expr.s test.

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ; RELAX: R_RISCV_ADD64 b
  ; RELAX: R_RISCV_SUB64 a
  ; NORELAX-NOT: R_RISCV
  call void asm sideeffect "a:\0Ab:\0A.dword b-a", ""()
  ret i32 0
}
