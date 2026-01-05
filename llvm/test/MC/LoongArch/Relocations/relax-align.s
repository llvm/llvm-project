## The file testing Nop insertion with R_LARCH_ALIGN for relaxation.

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s -o %t
# RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=INSTR
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.r
# RUN: llvm-objdump -d %t.r | FileCheck %s --check-prefixes=INSTR,RELAX-INSTR
# RUN: llvm-readobj -r %t.r | FileCheck %s --check-prefixes=RELOC,RELAX-RELOC

.text
call36 foo
# INSTR:      pcaddu18i $ra, 0
# INSTR-NEXT: jirl $ra, $ra, 0

## Not emit R_LARCH_ALIGN if alignment directive is less than or equal to
## minimum code alignment(a.k.a 4).
.p2align 2
.p2align 1
.p2align 0

## Not emit instructions if max emit bytes less than min nop size.
.p2align 4, , 2

## Not emit R_LARCH_ALIGN if alignment directive with specific padding value.
## The behavior is the same as GNU assembler.
break 1
.p2align 4, 1
# INSTR-NEXT: break 1
# INSTR-NEXT: 01 01 01 01

break 2
.p2align 4, 1, 12
# INSTR-NEXT:    break 2
# INSTR-COUNT-3: 01 01 01 01

break 3
.p2align 4
# INSTR-NEXT:    break 3
# INSTR-COUNT-3: nop

break 4
.p2align 5
.p2align 4
# INSTR-NEXT:          break 4
# INSTR-COUNT-3:       nop
# RELAX-INSTR-COUNT-7: nop

break 5
.p2align 4, , 11
# INSTR-NEXT: break 5
# RELAX-INSTR-COUNT-3: nop

break 6
## Not emit the third parameter.
.p2align 4, , 12
# INSTR-NEXT:       break 6
# INSTR-NEXT:       nop
# INSTR-NEXT:       nop
# RELAX-INSTR-NEXT: nop

ret
# INSNR-NEXT: ret

## Test the symbol index is different from .text.
.section .text2, "ax"
call36 foo
.p2align 4
.p2align 4, , 4
break 7

# RELOC:            Relocations [
# RELOC-NEXT:         Section ({{.*}}) .rela.text {
# RELOC-NEXT:           0x0 R_LARCH_CALL36 foo 0x0
# RELAX-RELOC-NEXT:     0x0 R_LARCH_RELAX - 0x0
# RELAX-RELOC-NEXT:     0x24 R_LARCH_ALIGN - 0xC
# RELAX-RELOC-NEXT:     0x34 R_LARCH_ALIGN - 0x1C
# RELAX-RELOC-NEXT:     0x50 R_LARCH_ALIGN - 0xC
# RELAX-RELOC-NEXT:     0x60 R_LARCH_ALIGN .Lla-relax-align0 0xB04
# RELAX-RELOC-NEXT:     0x70 R_LARCH_ALIGN - 0xC
# RELOC-NEXT:         }
# RELOC-NEXT:         Section ({{.*}}) .rela.text2 {
# RELOC-NEXT:           0x0 R_LARCH_CALL36 foo 0x0
# RELAX-RELOC-NEXT:     0x0 R_LARCH_RELAX - 0x0
# RELAX-RELOC-NEXT:     0x8 R_LARCH_ALIGN - 0xC
# RELAX-RELOC-NEXT:     0x14 R_LARCH_ALIGN .Lla-relax-align1 0x404
# RELOC-NEXT:         }
# RELOC-NEXT:       ]
