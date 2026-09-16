# RUN: llvm-mc %s -triple=riscv32 -mattr=+a,+zicfiss -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a,+zicfiss < %s \
# RUN:     | llvm-objdump --mattr=+a,+experimental-zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -defsym=RV64=1 -mattr=+a,+zicfiss -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-RV64,CHECK-ASM,CHECK-ASM-AND-OBJ-RV64,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -defsym=RV64=1 -mattr=+a,+zicfiss < %s \
# RUN:     | llvm-objdump --mattr=+a,+zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ-RV64,CHECK-ASM-AND-OBJ %s
#
# Zicfiss MOP-based instructions (sspopchk, ssrdp, sspush) only require Zimop,
# not Zicfiss. SSAMOSWAP still requires Zicfiss.
# (riscv-non-isa/riscv-elf-psabi-doc#474)
#
# RUN: not llvm-mc -triple riscv32 -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -defsym=RV64=1 -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT-RV64 %s

# CHECK-ASM-AND-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspopchk x1

# CHECK-ASM-AND-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspopchk ra

# CHECK-ASM-AND-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspopchk x5

# CHECK-ASM-AND-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspopchk t0

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspush x1

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspush ra

# CHECK-ASM-AND-OBJ: sspush t0
# CHECK-ASM: encoding: [0x73,0x40,0x50,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspush x5

# CHECK-ASM-AND-OBJ: sspush t0
# CHECK-ASM: encoding: [0x73,0x40,0x50,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
sspush t0

# CHECK-ASM-AND-OBJ: ssrdp ra
# CHECK-ASM: encoding: [0xf3,0x40,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
ssrdp ra

# sspopchk, sspush, and ssrdp share their encoding with the Zimop mop.r.28
# and mop.rr.7 instructions, differing only in which operands are fixed to
# x0. Verify that the generic Zimop mnemonics assemble to the same encoding,
# even though they are printed back using their own mnemonic (mop.r.28 /
# mop.rr.7 are separate instruction definitions, not aliases, so the
# assembly printer prints the mnemonic that was matched, while the
# disassembler still prefers the more specific sspopchk/ssrdp/sspush).

# CHECK-ASM: mop.r.28 zero, ra
# CHECK-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
mop.r.28 x0, x1

# CHECK-ASM: mop.r.28 zero, t0
# CHECK-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
mop.r.28 x0, x5

# CHECK-ASM: mop.r.28 ra, zero
# CHECK-OBJ: ssrdp ra
# CHECK-ASM: encoding: [0xf3,0x40,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
mop.r.28 x1, x0

# CHECK-ASM: mop.r.28 t0, zero
# CHECK-OBJ: ssrdp t0
# CHECK-ASM: encoding: [0xf3,0x42,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
mop.r.28 x5, x0

# CHECK-ASM: mop.rr.7 zero, zero, ra
# CHECK-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
mop.rr.7 x0, x0, x1

# CHECK-ASM: mop.rr.7 zero, zero, t0
# CHECK-OBJ: sspush t0
# CHECK-ASM: encoding: [0x73,0x40,0x50,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zimop' (May-Be-Operations)
mop.rr.7 x0, x0, x5

# CHECK-ASM-AND-OBJ: ssamoswap.w a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x48]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w a4, ra, (s0)

# CHECK-ASM-AND-OBJ: ssamoswap.w.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4c]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w.aq a4, ra, (s0)

# CHECK-ASM-AND-OBJ: ssamoswap.w.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w.rl a4, ra, (s0)

# CHECK-ASM-AND-OBJ: ssamoswap.w.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4e]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w.aqrl a4, ra, (s0)

.ifdef RV64
# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x48]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d a4, ra, (s0)

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.aq a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4c]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d.aq a4, ra, (s0)

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.rl a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4a]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d.rl a4, ra, (s0)

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.aqrl a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4e]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d.aqrl a4, ra, (s0)
.endif
