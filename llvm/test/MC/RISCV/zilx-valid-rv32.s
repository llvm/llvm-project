# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+experimental-zilx %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+experimental-zilx %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zilx --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# Unscaled indexed loads.

lxh a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxh a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x15,0xb6,0x90]

lxw a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxw a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x25,0xb6,0x90]

lxhu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxhu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x55,0xb6,0x90]

# Scaled indexed loads.

lxsb a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsb a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x05,0xb6,0xd0]

lxsh a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsh a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x15,0xb6,0xd0]

lxsw a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsw a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x25,0xb6,0xd0]

lxsbu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsbu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x45,0xb6,0xd0]

lxshu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxshu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x55,0xb6,0xd0]

# `lxb` and `lxbu` are assembler pseudoinstructions for `lxsb` and `lxsbu`.

lxb a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsb a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x05,0xb6,0xd0]

lxbu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsbu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x45,0xb6,0xd0]
