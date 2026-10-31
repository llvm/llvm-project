# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+experimental-zilx %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+experimental-zilx %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zilx --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# Unscaled indexed loads.

lxh a0, (a1), a2
# CHECK-INST: lxh a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x15,0xb6,0x90]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxw a0, (a1), a2
# CHECK-INST: lxw a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x25,0xb6,0x90]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxhu a0, (a1), a2
# CHECK-INST: lxhu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x55,0xb6,0x90]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

# Scaled indexed loads.

lxsb a0, (a1), a2
# CHECK-INST: lxsb a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x05,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsh a0, (a1), a2
# CHECK-INST: lxsh a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x15,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsw a0, (a1), a2
# CHECK-INST: lxsw a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x25,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsbu a0, (a1), a2
# CHECK-INST: lxsbu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x45,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxshu a0, (a1), a2
# CHECK-INST: lxshu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x55,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

# `lxb` and `lxbu` are assembler pseudoinstructions for `lxsb` and `lxsbu`.

lxb a0, (a1), a2
# CHECK-INST: lxsb a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x05,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxbu a0, (a1), a2
# CHECK-INST: lxsbu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x45,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}
