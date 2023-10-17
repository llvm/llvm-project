# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+m,+f,+d,+v noncanonicalized_arch.s -o noncanonicalized_arch.o
# RUN: not llvm-objdump -d noncanonicalized_arch.o 2>&1 | FileCheck %s -DFILE=noncanonicalized_arch.o --check-prefix=NONCANON

# RUN: llvm-mc -filetype=obj -triple=riscv64 invalid_arch.s -o invalid_arch.o
# RUN: not llvm-objdump -d invalid_arch.o 2>&1 | FileCheck %s -DFILE=invalid_arch.o --check-prefix=INVALID

# RUN: llvm-mc -filetype=obj -triple=riscv32 unknown_i_version.s -o unknown_i_version.o
# RUN: llvm-objdump -d unknown_i_version.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-I-VERSION

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zicbom unknown_ext_version.s -o unknown_ext_version.o
# RUN: llvm-objdump -d unknown_ext_version.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-EXT-VERSION

# RUN: llvm-mc -filetype=obj -triple=riscv64 unknown_ext_name.s -o unknown_ext_name.o
# RUN: llvm-objdump -d unknown_ext_name.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-EXT-NAME

#--- noncanonicalized_arch.s
# NONCANON: error: '[[FILE]]': arch string must begin with valid base ISA
# NONCANON-NOT: {{.}}
vsetvli a3, a2, e8, m8, tu, mu

.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64gcv"
.Lend:

#--- invalid_arch.s
# INVALID: error: '[[FILE]]': arch string must begin with valid base ISA
# INVALID-NOT: {{.}}
nop

.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "nonsense"
.Lend:

#--- unknown_i_version.s
# UNKNOWN-I-VERSION: nop
nop

.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv32i99p99"
.Lend:

#--- unknown_ext_version.s
# UNKNOWN-EXT-VERSION: cbo.clean (t0)
cbo.clean (t0)

.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv32i2p0_zicbom0p1"
.Lend:

#--- unknown_ext_name.s
# UNKNOWN-EXT-NAME: nop
nop

.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv32i2p0_zmadeup1p0_smadeup1p0_xmadeup1p0_sxmadeup1p0"
.Lend:
