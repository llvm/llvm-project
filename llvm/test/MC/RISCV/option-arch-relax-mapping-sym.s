# Test that .option arch emits ISA mapping symbols referenced by R_RISCV_RELAX,
# enabling the linker to determine per-region relaxation capabilities.
#
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax %s -o %t
# RUN: llvm-readobj -r --symbols %t | FileCheck %s

call no_arch_yet

# CHECK: R_RISCV_CALL_PLT no_arch_yet
# CHECK-NEXT: R_RISCV_RELAX $xrv64i2p1 0x0

.option arch, rv64imafdc
call with_c

# CHECK: R_RISCV_CALL_PLT with_c
# CHECK-NEXT: R_RISCV_RELAX $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0 0x0

.option arch, rv64imafd
call without_c

# CHECK: R_RISCV_CALL_PLT without_c
# CHECK-NEXT: R_RISCV_RELAX $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0 0x0

.option arch, rv64imafdc
call with_c_again

# CHECK: R_RISCV_CALL_PLT with_c_again
# CHECK-NEXT: R_RISCV_RELAX $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0 0x0

.option push
.option arch, rv64imafd
call inside_push_no_c

# CHECK: R_RISCV_CALL_PLT inside_push_no_c
# CHECK-NEXT: R_RISCV_RELAX $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0 0x0

.option pop
call after_pop_with_c

# CHECK: R_RISCV_CALL_PLT after_pop_with_c
# CHECK-NEXT: R_RISCV_RELAX $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0 0x0

# The initial base-ISA symbol appears at 0x0.
# CHECK: Name: $xrv64i2p1
# CHECK-NEXT: Value: 0x0
# The with-C symbol appears at 0x8 (start of second instruction pair).
# CHECK: Name: $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0
# CHECK-NEXT: Value: 0x8
# The without-C symbol appears at 0x10.
# CHECK: Name: $xrv64i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0
# CHECK-NEXT: Value: 0x10
