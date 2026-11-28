# RUN: llvm-mc -triple=sparc %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple=sparcv9 %s --defsym V9=1 | FileCheck %s --check-prefixes=ASM,ASM64
# RUN: llvm-mc -triple=sparc -filetype=obj %s | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -triple=sparcv9 -filetype=obj %s --defsym V9=1 | llvm-readobj -r - | FileCheck %s --check-prefixes=CHECK,CHECK64

# RUN: not llvm-mc -triple=sparc %s --defsym ERR0=1 2>&1 | FileCheck %s --check-prefix=ERR0 --implicit-check-not=error:
# RUN: not llvm-mc -triple=sparcv9 -filetype=obj %s --defsym ERR=1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

.globl g
g:
l:

# ASM:      .word %r_disp32(l)
# ASM-NEXT: .word %r_disp32(extern+4)
# ASM-NEXT: .word %r_disp32(g+8)
# ASM-NEXT: .byte %r_disp8(extern)
# ASM-NEXT: .half %r_disp16(extern+2)
# ASM-NEXT: .word %r_plt32(g)
# ASM-NEXT: .word %r_plt32(extern+4)
# ASM64:      .xword %r_disp64(extern)
# ASM64-NEXT: .xword %r_plt64(g)
# ASM64-NEXT: .xword %r_plt64(extern+8)

# CHECK:      Section ({{.*}}) .rela.data {
# CHECK-NEXT:   0x0 R_SPARC_DISP32 .text 0x0
# CHECK-NEXT:   0x4 R_SPARC_DISP32 extern 0x4
# CHECK-NEXT:   0x8 R_SPARC_DISP32 g 0x8
# CHECK-NEXT:   0xC R_SPARC_DISP8 extern 0x0
# CHECK-NEXT:   0xD R_SPARC_DISP16 extern 0x2
# CHECK-NEXT:   0xF R_SPARC_PLT32 g 0x0
# CHECK-NEXT:   0x13 R_SPARC_PLT32 extern 0x4
# CHECK64-NEXT: 0x17 R_SPARC_DISP64 extern 0x0
# CHECK64-NEXT: 0x1F R_SPARC_PLT64 g 0x0
# CHECK64-NEXT: 0x27 R_SPARC_PLT64 extern 0x8
# CHECK-NEXT: }
.data
.word %r_disp32(l)
.word %r_disp32(extern + 4), %r_disp32(g + 8)
.byte %r_disp8(extern)
.half %r_disp16(extern + 2)
.word %r_plt32(g), %r_plt32(extern + 4)

.ifdef V9
.xword %r_disp64(extern)
.xword %r_plt64(g), %r_plt64(extern + 8)
.endif

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
# ERR: [[#@LINE+1]]:8: error: %r_disp8 can only be used in a .byte directive
.word %r_disp8(g)
# ERR: [[#@LINE+1]]:8: error: %r_disp16 can only be used in a .half directive
.word %r_disp16(g)
# ERR: [[#@LINE+1]]:8: error: %r_disp64 can only be used in a .xword directive
.word %r_disp64(g)
# ERR: [[#@LINE+1]]:9: error: %r_plt32 can only be used in a .word directive
.xword %r_plt32(g)
# ERR: [[#@LINE+1]]:8: error: %r_plt64 can only be used in a .xword directive
.word %r_plt64(g)
.endif
