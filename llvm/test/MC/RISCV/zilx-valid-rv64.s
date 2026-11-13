# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zilx %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zilx %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zilx --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# Unscaled indexed loads (RV64-only widths).

lxd a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxd a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x35,0xb6,0x90]

lxwu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxwu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x65,0xb6,0x90]

# Scaled indexed loads (RV64-only widths).

lxsd a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsd a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x35,0xb6,0xd0]

lxswu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxswu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x65,0xb6,0xd0]

# Scaled indexed loads with a zero-extended 32-bit index (RV64-only).

lxsuwb a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsuwb a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x05,0xb6,0xf0]

lxsuwh a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsuwh a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x15,0xb6,0xf0]

lxsuww a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsuww a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x25,0xb6,0xf0]

lxsuwd a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsuwd a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x35,0xb6,0xf0]

lxsuwbu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsuwbu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x45,0xb6,0xf0]

lxsuwhu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsuwhu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x55,0xb6,0xf0]

lxsuwwu a0, (a1), a2 # expected-error {{instruction requires the following: 'Zilx' (Indexed Integer Load Instructions)}}
# CHECK-INST: lxsuwwu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x65,0xb6,0xf0]
