## Target specific relocation support is tested in MC/$target/*reloc-directive*.s
# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# ASM:      .Ltmp0:
# ASM-NEXT:  .reloc .Ltmp0+3-2, R_X86_64_NONE, foo
# ASM-NEXT: .Ltmp1:
# ASM-NEXT:  .reloc .Ltmp1-1, R_X86_64_NONE, foo
# ASM-NEXT: .Ltmp2:
# ASM-NEXT:  .reloc 2+.Ltmp2, R_X86_64_NONE, local

# CHECK:      0x2 R_X86_64_NONE foo 0x0
# CHECK-NEXT: 0x0 R_X86_64_NONE foo 0x0
# CHECK-NEXT: 0x3 R_X86_64_NONE local 0x0
# CHECK-NEXT: 0x1 R_X86_64_NONE unused 0x0
# CHECK-NEXT: 0x4 R_X86_64_NONE data 0x1

# CHECK:      .rela.my {
# CHECK:        0x0 R_X86_64_NONE foo 0x0
# CHECK-NEXT:   0x4 R_X86_64_NONE foo 0x0
# CHECK-NEXT:   0x8 R_X86_64_NONE foo 0x0
# CHECK-NEXT: }

.text
.globl foo
foo:
local:
  ret
  .reloc .+3-2, R_X86_64_NONE, foo
  .reloc .-1, R_X86_64_NONE, foo
  .reloc 2+., R_X86_64_NONE, local
  .reloc ., BFD_RELOC_NONE, unused
  .space 3

.data
.globl data
data:
  .reloc 1+foo+3, R_X86_64_NONE, data+1
  .long 0

## Constant offsets are relative to the section start.
.section .my
.word 0
.reloc 0, BFD_RELOC_NONE, foo
.word 0
.p2align 3
.reloc 2+2, BFD_RELOC_NONE, foo
.p2align 4
.reloc 8, BFD_RELOC_NONE, foo

.text
.globl a, b
a: ret
b: ret
x: ret
y: ret

# RUN: not llvm-mc -filetype=obj -triple=x86_64 --defsym=PARSE=1 %s 2>&1 | FileCheck %s --check-prefix=PARSE
# RUN: not llvm-mc -filetype=obj -triple=x86_64 --defsym=ERR=1 %s 2>&1 | FileCheck %s --check-prefix=ERR

.ifdef PARSE
# PARSE: {{.*}}.s:[[#@LINE+1]]:10: error: expected comma
.reloc 0 R_X86_64_NONE, a

# PARSE: {{.*}}.s:[[#@LINE+1]]:8: error: directional label undefined
.reloc 1f, R_X86_64_NONE, a
.endif

.ifdef ERR
.reloc -1, R_X86_64_NONE, a
# ERR: {{.*}}.s:[[#@LINE+1]]:9: error: .reloc offset is not relocatable
.reloc 2*., R_X86_64_NONE, a
# ERR: {{.*}}.s:[[#@LINE+1]]:9: error: .reloc offset is not relocatable
.reloc a+a, R_X86_64_NONE, a
# ERR: {{.*}}.s:[[#@LINE+1]]:9: error: .reloc offset is not relative to a section
.reloc b-a, R_X86_64_NONE, a
# ERR: {{.*}}.s:[[#@LINE+1]]:9: error: .reloc offset is not relative to a section
.reloc x-x, R_X86_64_NONE, a

.endif
