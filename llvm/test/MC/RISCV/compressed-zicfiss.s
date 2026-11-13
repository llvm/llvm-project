# RUN: llvm-mc %s -triple=riscv32 -mattr=+zicfiss,+zcmop -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zicfiss,+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+zicfiss,+zcmop -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zicfiss,+zcmop -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zicfiss,+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+zicfiss,+zcmop -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# Compressed Zicfiss instructions only require Zcmop (and Zimop for
# uncompressed forms), not Zicfiss (riscv-non-isa/riscv-elf-psabi-doc#474).
#
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zcmop,+zimop -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zcmop,+zimop < %s \
# RUN:     | llvm-objdump --mattr=+zcmop,+zimop -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zcmop,+zimop -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zcmop,+zimop < %s \
# RUN:     | llvm-objdump --mattr=+zcmop,+zimop -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -M no-aliases -show-encoding < %s

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
sspopchk x5 # expected-error {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
sspopchk t0 # expected-error {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
sspush x1 # expected-error {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
sspush ra # expected-error {{instruction requires the following: 'Zimop' (May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
c.sspush x1 # expected-error {{instruction requires the following: 'Zcmop' (Compressed May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
c.sspush ra # expected-error {{instruction requires the following: 'Zcmop' (Compressed May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
c.sspopchk x5 # expected-error {{instruction requires the following: 'Zcmop' (Compressed May-Be-Operations)}}

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
c.sspopchk t0 # expected-error {{instruction requires the following: 'Zcmop' (Compressed May-Be-Operations)}}
