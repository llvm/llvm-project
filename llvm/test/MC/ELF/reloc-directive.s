## Target specific relocation support is tested in MC/$target/*reloc-directive*.s
# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# ASM:      .Ltmp0:
# ASM-NEXT:  .reloc (.Ltmp0+3)-2, R_X86_64_NONE, foo
# ASM-NEXT: .Ltmp1:
# ASM-NEXT:  .reloc .Ltmp1-1, R_X86_64_NONE, foo
# ASM-NEXT: .Ltmp2:
# ASM-NEXT:  .reloc 2+.Ltmp2, R_X86_64_NONE, foo
# ASM-NEXT:  .reloc (1+foo)+3, R_X86_64_NONE, data+1
# ASM-NEXT: .Ltmp3:
# ASM-NEXT:  .reloc .Ltmp3, BFD_RELOC_NONE, unused

# CHECK:      0x2 R_X86_64_NONE foo 0x0
# CHECK-NEXT: 0x0 R_X86_64_NONE foo 0x0
# CHECK-NEXT: 0x3 R_X86_64_NONE foo 0x0
# CHECK-NEXT: 0x4 R_X86_64_NONE data 0x1
# CHECK-NEXT: 0x1 R_X86_64_NONE unused 0x0

.text
.globl foo
foo:
  ret
  .reloc .+3-2, R_X86_64_NONE, foo
  .reloc .-1, R_X86_64_NONE, foo
  .reloc 2+., R_X86_64_NONE, foo
  .reloc 1+foo+3, R_X86_64_NONE, data+1
  .reloc ., BFD_RELOC_NONE, unused

.data
.globl data
data:
  .long 0

# RUN: not llvm-mc -filetype=obj -triple=x86_64 --defsym=ERR=1 %s 2>&1 | FileCheck %s --check-prefix=ERR

.ifdef ERR
.text
.globl a, b
a: ret
b: ret
x: ret
y: ret

# ERR: {{.*}}.s:[[#@LINE+1]]:10: error: expected comma
.reloc 0 R_X86_64_NONE, a

# ERR: {{.*}}.s:[[#@LINE+1]]:8: error: .reloc offset is negative
.reloc -1, R_X86_64_NONE, a
# ERR: {{.*}}.s:[[#@LINE+1]]:8: error: .reloc offset is not relocatable
.reloc 2*., R_X86_64_NONE, a
# ERR: {{.*}}.s:[[#@LINE+1]]:8: error: .reloc offset is not relocatable
.reloc a+a, R_X86_64_NONE, a
## GNU as accepts a-a but rejects b-a.
# ERR: {{.*}}.s:[[#@LINE+1]]:8: error: .reloc offset is not representable
.reloc a-a, R_X86_64_NONE, a
## TODO GNU as accepts x-x and y-x.
# ERR: {{.*}}.s:[[#@LINE+1]]:8: error: .reloc offset is not representable
.reloc x-x, R_X86_64_NONE, a

# ERR: {{.*}}.s:[[#@LINE+1]]:8: error: directional label undefined
.reloc 1f, R_X86_64_NONE, a
.endif
