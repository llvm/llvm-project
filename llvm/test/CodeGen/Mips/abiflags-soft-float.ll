; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o tmp.o
; RUN: llvm-readobj -A tmp.o | FileCheck %s -check-prefix=OBJ
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | \
; RUN: FileCheck %s -check-prefix=ASM

; OBJ: FP ABI: Soft float
; ASM: .module	softfloat 

define dso_local void @asm_is_null() "use-soft-float"="true" {
  call void asm sideeffect "", ""()
  ret void
}
