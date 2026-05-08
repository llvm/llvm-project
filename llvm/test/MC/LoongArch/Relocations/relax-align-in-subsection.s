## The file testing R_LARCH_ALIGN emitting when linker-relaxation enabled.

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.n
# RUN: llvm-objdump -dr %t.n | FileCheck %s --check-prefix=RELAX
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax --defsym FILL=1 %s -o %t.f
# RUN: llvm-objdump -dr %t.f | FileCheck %s --check-prefixes=RELAX,ALIGN

# ALIGN:         nop
# ALIGN-NEXT:        R_LARCH_ALIGN *ABS*+0x1c
# ALIGN-COUNT-6: nop
# RELAX:         ret
# RELAX:         pcaddu18i $ra, 0
# RELAX-NEXT:        R_LARCH_CALL36 f
# RELAX-NEXT:        R_LARCH_RELAX *ABS*
# RELAX-NEXT:    jirl $ra, $ra, 0

.text
.option push
.option norelax
## When FILL is defined, the order of Alignment directive in this lower-numbered
## subsection will be larger, and even larger than the section order of the first
## linker-relaxable call36 instruction. It should conservatively be treated as
## linker-relaxable even has norelax.
.ifdef FILL
  .space 0
.endif
.p2align 5
foo:
  ret
.option pop

.text 1
  .space 0
bar:
  call36 f
