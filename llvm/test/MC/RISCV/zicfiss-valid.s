# RUN: llvm-mc %s -triple=riscv32 -mattr=+a,+zicfiss -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a,+zicfiss < %s \
# RUN:     | llvm-objdump --mattr=+a,+zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -defsym=RV64=1 -mattr=+a,+zicfiss -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-RV64,CHECK-ASM,CHECK-ASM-AND-OBJ-RV64,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -defsym=RV64=1 -mattr=+a,+zicfiss < %s \
# RUN:     | llvm-objdump --mattr=+a,+zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ-RV64,CHECK-ASM-AND-OBJ %s
#
# Zicfiss MOP-based instructions (sspopchk, ssrdp, sspush) only require Zimop,
# not Zicfiss. SSAMOSWAP still requires Zicfiss.
# (riscv-non-isa/riscv-elf-psabi-doc#474)
#
# RUN: llvm-mc --verify=no-ext -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify=no-ext-rv64 -triple riscv64 -defsym=RV64=1 -M no-aliases -show-encoding < %s

# CHECK-ASM-AND-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
sspopchk x1
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
sspopchk ra
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
sspopchk x5
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
sspopchk t0
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
sspush x1
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
sspush ra
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspush t0
# CHECK-ASM: encoding: [0x73,0x40,0x50,0xce]
sspush x5
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspush t0
# CHECK-ASM: encoding: [0x73,0x40,0x50,0xce]
sspush t0
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: ssrdp ra
# CHECK-ASM: encoding: [0xf3,0x40,0xc0,0xcd]
ssrdp ra
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# sspopchk, sspush, and ssrdp share their encoding with the Zimop mop.r.28
# and mop.rr.7 instructions, differing only in which operands are fixed to
# x0. Verify that the generic Zimop mnemonics assemble to the same encoding,
# even though they are printed back using their own mnemonic (mop.r.28 /
# mop.rr.7 are separate instruction definitions, not aliases, so the
# assembly printer prints the mnemonic that was matched, while the
# disassembler still prefers the more specific sspopchk/ssrdp/sspush).

# CHECK-ASM-AND-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
mop.r.28 x0, x1
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
mop.r.28 x0, x5
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: ssrdp ra
# CHECK-ASM: encoding: [0xf3,0x40,0xc0,0xcd]
mop.r.28 x1, x0
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: ssrdp t0
# CHECK-ASM: encoding: [0xf3,0x42,0xc0,0xcd]
mop.r.28 x5, x0
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
mop.rr.7 x0, x0, x1
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: sspush t0
# CHECK-ASM: encoding: [0x73,0x40,0x50,0xce]
mop.rr.7 x0, x0, x5
# no-ext-error@-1 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: ssamoswap.w a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x48]
ssamoswap.w a4, ra, (s0)
# no-ext-error@-1 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}

# CHECK-ASM-AND-OBJ: ssamoswap.w.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4c]
ssamoswap.w.aq a4, ra, (s0)
# no-ext-error@-1 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}

# CHECK-ASM-AND-OBJ: ssamoswap.w.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4a]
ssamoswap.w.rl a4, ra, (s0)
# no-ext-error@-1 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}

# CHECK-ASM-AND-OBJ: ssamoswap.w.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4e]
ssamoswap.w.aqrl a4, ra, (s0)
# no-ext-error@-1 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}
# no-ext-rv64-error@-2 {{instruction requires the following: 'Zicfiss' (Shadow stack)}}

.ifdef RV64
# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x48]
ssamoswap.d a4, ra, (s0) # no-ext-rv64-error {{instruction requires the following: 'Zicfiss' (Shadow stack)}}

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.aq a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4c]
ssamoswap.d.aq a4, ra, (s0) # no-ext-rv64-error {{instruction requires the following: 'Zicfiss' (Shadow stack)}}

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.rl a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4a]
ssamoswap.d.rl a4, ra, (s0) # no-ext-rv64-error {{instruction requires the following: 'Zicfiss' (Shadow stack)}}

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.aqrl a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4e]
ssamoswap.d.aqrl a4, ra, (s0) # no-ext-rv64-error {{instruction requires the following: 'Zicfiss' (Shadow stack)}}
.endif
