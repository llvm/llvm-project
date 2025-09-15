# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=LA32-RELAX-RELOC %s
# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=LA32-NORELAX-RELOC %s

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax --defsym=LA64=1 < %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=LA64-RELAX-RELOC %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax --defsym=LA64=1 < %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=LA64-NORELAX-RELOC %s

.long foo

.ifndef LA64

lu12i.w $a0, %le_hi20_r(foo)
# LA32-NORELAX-RELOC: R_LARCH_TLS_LE_HI20_R foo 0x0
# LA32-NORELAX-RELOC-NOT: R_LARCH_RELAX - 0x0
# LA32-RELAX-RELOC: R_LARCH_TLS_LE_HI20_R foo 0x0
# LA32-RELAX-RELOC: R_LARCH_RELAX - 0x0

add.w $a0, $a0, $tp, %le_add_r(foo)
# LA32-NORELAX-RELOC: R_LARCH_TLS_LE_ADD_R foo 0x0
# LA32-NORELAX-RELOC-NOT: R_LARCH_RELAX - 0x0
# LA32-RELAX-RELOC: R_LARCH_TLS_LE_ADD_R foo 0x0
# LA32-RELAX-RELOC: R_LARCH_RELAX - 0x0

addi.w $a0, $a0, %le_lo12_r(foo)
# LA32-NORELAX-RELOC: R_LARCH_TLS_LE_LO12_R foo 0x0
# LA32-NORELAX-RELOC-NOT: R_LARCH_RELAX - 0x0
# LA32-RELAX-RELOC: R_LARCH_TLS_LE_LO12_R foo 0x0
# LA32-RELAX-RELOC: R_LARCH_RELAX - 0x0

.else

lu12i.w $a0, %le_hi20_r(foo)
# LA64-NORELAX-RELOC: R_LARCH_TLS_LE_HI20_R foo 0x0
# LA64-NORELAX-RELOC-NOT: R_LARCH_RELAX - 0x0
# LA64-RELAX-RELOC: R_LARCH_TLS_LE_HI20_R foo 0x0
# LA64-RELAX-RELOC: R_LARCH_RELAX - 0x0

add.d $a0, $a0, $tp, %le_add_r(foo)
# LA64-NORELAX-RELOC: R_LARCH_TLS_LE_ADD_R foo 0x0
# LA64-NORELAX-RELOC-NOT: R_LARCH_RELAX - 0x0
# LA64-RELAX-RELOC: R_LARCH_TLS_LE_ADD_R foo 0x0
# LA64-RELAX-RELOC: R_LARCH_RELAX - 0x0

addi.d $a0, $a0, %le_lo12_r(foo)
# LA64-NORELAX-RELOC: R_LARCH_TLS_LE_LO12_R foo 0x0
# LA64-NORELAX-RELOC-NOT: R_LARCH_RELAX - 0x0
# LA64-RELAX-RELOC: R_LARCH_TLS_LE_LO12_R foo 0x0
# LA64-RELAX-RELOC: R_LARCH_RELAX - 0x0

.endif

