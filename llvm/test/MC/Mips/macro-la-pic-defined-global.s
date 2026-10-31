# RUN: llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   -filetype=obj -o - | llvm-readobj -r - | FileCheck --check-prefix=O32 %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 -mattr=+xgot \
# RUN:   -filetype=obj -o - | llvm-readobj -r - | FileCheck --check-prefix=XGOT %s
# RUN: llvm-mc %s -triple=mipsn32 -mcpu=mips64r2 \
# RUN:   -filetype=obj -o - | llvm-readobj -r - | FileCheck --check-prefix=N32 %s

.option pic2

.data
.globl global_symbol
global_symbol:
  .word 0

.local local_symbol
local_symbol:
  .word 0

.text
la $5, global_symbol
la $25, global_symbol
la $6, local_symbol

# O32:      Section {{.*}} .rel.text {
# O32-NEXT:   0x0 R_MIPS_GOT16 global_symbol
# O32-NEXT:   0x4 R_MIPS_CALL16 global_symbol
# O32-NEXT:   0x8 R_MIPS_GOT16 .data
# O32-NEXT:   0xC R_MIPS_LO16 .data
# O32-NEXT: }

# XGOT:      Section {{.*}} .rel.text {
# XGOT-NEXT:   0x0 R_MIPS_GOT_HI16 global_symbol
# XGOT-NEXT:   0x8 R_MIPS_GOT_LO16 global_symbol
# XGOT-NEXT:   0xC R_MIPS_CALL_HI16 global_symbol
# XGOT-NEXT:   0x14 R_MIPS_CALL_LO16 global_symbol
# XGOT-NEXT:   0x18 R_MIPS_GOT16 .data
# XGOT-NEXT:   0x1C R_MIPS_LO16 .data
# XGOT-NEXT: }

# N32:      Section {{.*}} .rela.text {
# N32-NEXT:   0x0 R_MIPS_GOT_DISP global_symbol 0x0
# N32-NEXT:   0x4 R_MIPS_CALL16 global_symbol 0x0
# N32-NEXT:   0x8 R_MIPS_GOT_DISP local_symbol 0x0
# N32-NEXT: }
