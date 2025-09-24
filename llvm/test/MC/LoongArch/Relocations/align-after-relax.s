## The file testing R_LARCH_ALIGN emitting when linker-relaxation enabled.

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s -o %t.n
# RUN: llvm-objdump -dr %t.n | FileCheck %s --check-prefix=NORELAX
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.r
# RUN: llvm-objdump -dr %t.r | FileCheck %s --check-prefix=RELAX

# NORELAX:         pcaddu18i $ra, 0
# NORELAX-NEXT:        R_LARCH_CALL36 f
# NORELAX-NEXT:    jirl $ra, $ra, 0
# NORELAX-COUNT-6: nop
# NORELAX:         pcaddu18i $ra, 0
# NORELAX-NEXT:        R_LARCH_CALL36 f
# NORELAX-NEXT:    jirl $ra, $ra, 0
# NORELAX-COUNT-6: nop
# NORELAX:         pcaddu18i $ra, 0
# NORELAX-NEXT:        R_LARCH_CALL36 f
# NORELAX-NEXT:    jirl $ra, $ra, 0

# RELAX:         pcaddu18i $ra, 0
# RELAX-NEXT:        R_LARCH_CALL36 f
# RELAX-NEXT:        R_LARCH_RELAX *ABS*
# RELAX-NEXT:    jirl $ra, $ra, 0
# RELAX-NEXT:    nop
# RELAX-NEXT:        R_LARCH_ALIGN *ABS*+0x1c
# RELAX-COUNT-6: nop
# RELAX:         pcaddu18i $ra, 0
# RELAX-NEXT:        R_LARCH_CALL36 f
# RELAX-NEXT:        R_LARCH_RELAX *ABS*
# RELAX-NEXT:    jirl $ra, $ra, 0
# RELAX-NEXT:    nop
# RELAX-NEXT:        R_LARCH_ALIGN *ABS*+0x1c
# RELAX-COUNT-6: nop
# RELAX:         pcaddu18i $ra, 0
# RELAX-NEXT:        R_LARCH_CALL36 f
# RELAX-NEXT:    jirl $ra, $ra, 0

.text
## No R_LARCH_ALIGN before the first linker-relaxable instruction.
.p2align 5
foo:
  call36 f

## R_LARCH_ALIGN is required after the first linker-relaxable instruction.
.p2align 5
bar:
  call36 f

.option push
.option norelax
## R_LARCH_ALIGN is required even if norelax, because it is after a
## linker-relaxable instruction. No R_LARCH_RELAX for call36 because
## of the norelax.
.p2align 5
baz:
  call36 f
.option pop
