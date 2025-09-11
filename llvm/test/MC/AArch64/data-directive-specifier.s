# RUN: llvm-mc -triple=aarch64 %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple=aarch64 -filetype=obj %s | llvm-readobj -r - | FileCheck %s

# RUN: not llvm-mc -triple=aarch64 %s --defsym ERR0=1 2>&1 | FileCheck %s --check-prefix=ERR0 --implicit-check-not=error:
# RUN: not llvm-mc -triple=aarch64 -filetype=obj %s --defsym ERR=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

.globl g
g:
l:

# ASM: .word %pltpcrel(l)
# CHECK:      Section ({{.*}}) .rela.data {
# CHECK-NEXT:   0x0 R_AARCH64_PLT32 l 0x0
# CHECK-NEXT:   0x4 R_AARCH64_PLT32 extern 0x4
# CHECK-NEXT:   0x8 R_AARCH64_PLT32 g 0x8
# CHECK-NEXT: }
.data
.word %pltpcrel(l)
.word %pltpcrel(extern + 4), %pltpcrel(g + 8)

# ASM: .word %gotpcrel(data1)
# CHECK:      Section ({{.*}}) .rela.data1 {
# CHECK-NEXT:   0x0 R_AARCH64_GOTPCREL32 data1 0x0
# CHECK-NEXT:   0x4 R_AARCH64_GOTPCREL32 extern 0x4
# CHECK-NEXT:   0x8 R_AARCH64_GOTPCREL32 extern 0xFFFFFFFFFFFFFFFB
# CHECK-NEXT: }
.section .data1,"aw"
data1:
.word %gotpcrel(data1)
.word %gotpcrel(extern+4), %gotpcrel(extern-5)

.ifdef ERR0
# ERR0: [[#@LINE+1]]:8: error: invalid relocation specifier
.word %xxx(l)

# ERR0: [[#@LINE+1]]:17: error: expected '('
.word %pltpcrel l

# ERR0: [[#@LINE+2]]:14: error: unknown token in expression
# ERR0: [[#@LINE+1]]:14: error: invalid operand
ldr w0, [x1, %pltpcrel(g)]
.endif

.ifdef ERR
# ERR: [[#@LINE+1]]:8: error: %pltpcrel can only be used in a .word directive
.quad %pltpcrel(g)

# ERR: [[#@LINE+1]]:8: error: expected relocatable expression
.word %pltpcrel(g-.)

# ERR: [[#@LINE+1]]:8: error: expected relocatable expression
.word %pltpcrel(extern - und)

# ERR: [[#@LINE+1]]:8: error: %gotpcrel can only be used in a .word directive
.quad %gotpcrel(g)

# ERR: [[#@LINE+1]]:8: error: expected relocatable expression
.word %gotpcrel(extern - .)

# ERR: [[#@LINE+1]]:8: error: expected relocatable expression
.word %gotpcrel(extern - und)
.endif
