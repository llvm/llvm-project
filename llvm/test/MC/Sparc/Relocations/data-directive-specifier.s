# RUN: llvm-mc -triple=sparc %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple=sparc -filetype=obj %s | llvm-readobj -r - | FileCheck %s

# RUN: not llvm-mc -triple=sparc %s --defsym ERR0=1 2>&1 | FileCheck %s --check-prefix=ERR0 --implicit-check-not=error:
# RUN: not llvm-mc -triple=sparc -filetype=obj %s --defsym ERR=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

.globl g
g:
l:

# ASM:      .word %r_disp32(l)
# ASM-NEXT: .word %r_disp32(extern+4)
# ASM-NEXT: .word %r_disp32(g+8)

# CHECK:      Section ({{.*}}) .rela.data {
# CHECK-NEXT:   0x0 R_SPARC_DISP32 .text 0x0
# CHECK-NEXT:   0x4 R_SPARC_DISP32 extern 0x4
# CHECK-NEXT:   0x8 R_SPARC_DISP32 g 0x8
# CHECK-NEXT: }
.data
.word %r_disp32(l)
.word %r_disp32(extern + 4), %r_disp32(g + 8)

.ifdef ERR0
# ERR0: [[#@LINE+1]]:8: error: invalid relocation specifier
.word %hi(g)

# ERR0: [[#@LINE+1]]:17: error: expected '('
.word %r_disp32 g

# ERR0: [[#@LINE+2]]:8: error: invalid relocation specifier
# ERR0: [[#@LINE+1]]:8: error: unexpected token
sethi %r_disp32(g), %g1
.endif

.ifdef ERR
# ERR: [[#@LINE+1]]:8: error: %r_disp32 can only be used in a .word directive
.quad %r_disp32(g)
.endif
