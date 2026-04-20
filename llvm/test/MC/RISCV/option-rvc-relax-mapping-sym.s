# Test .option rvc/.option norvc ISA mapping symbol emission and R_RISCV_RELAX
# symbol references across four combinations of initial relax/rvc state
# (RISC-V ELF PSABI PR #393).
#
# Each RUN assembles the same source with a different -mattr combination.
#   RELAX-NORVC   (+relax, no C) - baseline RVC off, relaxation on
#   RELAX-RVC     (+relax, +c)   - baseline RVC on,  relaxation on
#   NORELAX-NORVC (no mattr)     - baseline RVC off, relaxation off
#   NORELAX-RVC   (+c)           - baseline RVC on,  relaxation off
#
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax     %s \
# RUN:   -o %t.relax_norvc
# RUN: llvm-readobj -r --symbols %t.relax_norvc \
# RUN:   | FileCheck --check-prefix=RELAX-NORVC %s
#
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax,+c  %s \
# RUN:   -o %t.relax_rvc
# RUN: llvm-readobj -r --symbols %t.relax_rvc \
# RUN:   | FileCheck --check-prefix=RELAX-RVC %s
#
# RUN: llvm-mc -filetype=obj -triple riscv64                    %s \
# RUN:   -o %t.norelax_norvc
# RUN: llvm-readobj -r --symbols %t.norelax_norvc \
# RUN:   | FileCheck --check-prefix=NORELAX-NORVC %s
#
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c          %s \
# RUN:   -o %t.norelax_rvc
# RUN: llvm-readobj -r --symbols %t.norelax_rvc \
# RUN:   | FileCheck --check-prefix=NORELAX-RVC %s

# An initial "$x<ISAString>" symbol is emitted before the very first
# instruction so R_RISCV_RELAX always has a concrete ISA reference.
nop
call before_option

# RELAX-NORVC:      R_RISCV_CALL_PLT before_option
# RELAX-NORVC-NEXT: R_RISCV_RELAX $xrv64i2p1 0x0
# RELAX-RVC:        R_RISCV_CALL_PLT before_option
# RELAX-RVC-NEXT:   R_RISCV_RELAX $xrv64i2p1_c2p0_zca1p0 0x0
# NORELAX-NORVC:    R_RISCV_CALL_PLT before_option
# NORELAX-NORVC-NOT: R_RISCV_RELAX
# NORELAX-RVC:      R_RISCV_CALL_PLT before_option
# NORELAX-RVC-NOT:  R_RISCV_RELAX

# .option rvc enables RVC; an ISA mapping symbol is emitted (deduplication:
# only when the ISA actually changes).  When starting with +c the directive
# is a no-op so the ISA sym and R_RISCV_RELAX target remain unchanged.
.option rvc
call after_rvc

# RELAX-NORVC:      R_RISCV_CALL_PLT after_rvc
# RELAX-NORVC-NEXT: R_RISCV_RELAX $xrv64i2p1_c2p0_zca1p0 0x0
# RELAX-RVC:        R_RISCV_CALL_PLT after_rvc
# RELAX-RVC-NEXT:   R_RISCV_RELAX $xrv64i2p1_c2p0_zca1p0 0x0
# NORELAX-NORVC:    R_RISCV_CALL_PLT after_rvc
# NORELAX-NORVC-NOT: R_RISCV_RELAX
# NORELAX-RVC:      R_RISCV_CALL_PLT after_rvc
# NORELAX-RVC-NOT:  R_RISCV_RELAX

# .option norvc disables RVC; a new ISA mapping symbol without C is emitted.
.option norvc
call after_norvc

# RELAX-NORVC:      R_RISCV_CALL_PLT after_norvc
# RELAX-NORVC-NEXT: R_RISCV_RELAX $xrv64i2p1 0x0
# RELAX-RVC:        R_RISCV_CALL_PLT after_norvc
# RELAX-RVC-NEXT:   R_RISCV_RELAX $xrv64i2p1 0x0
# NORELAX-NORVC:    R_RISCV_CALL_PLT after_norvc
# NORELAX-NORVC-NOT: R_RISCV_RELAX
# NORELAX-RVC:      R_RISCV_CALL_PLT after_norvc
# NORELAX-RVC-NOT:  R_RISCV_RELAX

# Verify ISA mapping symbols at the correct offsets.  In all cases the initial
# ISA symbol is emitted at 0x0.
#
# Layout with RVC off (nop = 4 bytes):
#   0x00  initial sym + nop,  0x04  call before_option,
#   0x0C  .option rvc (new sym),  call after_rvc,
#   0x14  .option norvc (new sym),  call after_norvc
#
# RELAX-NORVC:       Name: $xrv64i2p1
# RELAX-NORVC-NEXT:  Value: 0x0
# RELAX-NORVC:       Name: $xrv64i2p1_c2p0_zca1p0
# RELAX-NORVC-NEXT:  Value: 0xC
# RELAX-NORVC:       Name: $xrv64i2p1
# RELAX-NORVC-NEXT:  Value: 0x14
# NORELAX-NORVC:     Name: $xrv64i2p1
# NORELAX-NORVC-NEXT: Value: 0x0
# NORELAX-NORVC:     Name: $xrv64i2p1_c2p0_zca1p0
# NORELAX-NORVC-NEXT: Value: 0xC
# NORELAX-NORVC:     Name: $xrv64i2p1
# NORELAX-NORVC-NEXT: Value: 0x14
#
# Layout with RVC on (nop = c.nop = 2 bytes):
#   0x00  initial sym + c.nop,  0x02  call before_option,
#   0x0A  .option rvc (no-op),  call after_rvc,
#   0x12  .option norvc (new sym),  call after_norvc
#
# RELAX-RVC:        Name: $xrv64i2p1_c2p0_zca1p0
# RELAX-RVC-NEXT:   Value: 0x0
# RELAX-RVC:        Name: $xrv64i2p1
# RELAX-RVC-NEXT:   Value: 0x12
# NORELAX-RVC:      Name: $xrv64i2p1_c2p0_zca1p0
# NORELAX-RVC-NEXT: Value: 0x0
# NORELAX-RVC:      Name: $xrv64i2p1
# NORELAX-RVC-NEXT: Value: 0x12
