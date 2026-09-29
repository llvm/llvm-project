## For a target defined in the same section, create a relocation if the
## symbol is not local, otherwise resolve the fixup statically.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck  %s

# RELOC:      .rela.text {
# RELOC-NEXT:   0x5 R_X86_64_PLT32 global 0xFFFFFFFFFFFFFFFC
# RELOC-NEXT:   0xA R_X86_64_PLT32 weak 0xFFFFFFFFFFFFFFFC
# RELOC-NEXT:   0x19 R_X86_64_PLT32 global 0xFFFFFFFFFFFFFFFC
# RELOC-NEXT:   0x1E R_X86_64_PLT32 weak 0xFFFFFFFFFFFFFFFC
# RELOC-NEXT:   0x23 R_X86_64_PLT32 ifunc 0xFFFFFFFFFFFFFFFC
# RELOC-NEXT: }

# CHECK:      0: jmp
# CHECK-NEXT: 2: jmp
# CHECK-NEXT: 4: jmp
# CHECK-NEXT: 9: jmp
# CHECK-NEXT: e: callq
# CHECK-NEXT: 13: callq
# CHECK-NEXT: 18: callq
# CHECK-NEXT: 1d: callq
# CHECK-NEXT: 22: callq
# CHECK-NEXT: 27: retq

.globl global
.weak weak
.type ifunc,@gnu_indirect_function
global:
weak:
local:
.set var,global
ifunc:
  jmp var
  jmp local
  jmp global
  jmp weak

  call var
  call local
  call global
  call weak

  call ifunc
  ret
