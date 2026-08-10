# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zilx %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zilx %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zilx --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

# Unscaled indexed loads (RV64-only widths).

lxd a0, (a1), a2
# CHECK-INST: lxd a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x35,0xb6,0x90]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxwu a0, (a1), a2
# CHECK-INST: lxwu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x65,0xb6,0x90]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

# Scaled indexed loads (RV64-only widths).

lxsd a0, (a1), a2
# CHECK-INST: lxsd a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x35,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxswu a0, (a1), a2
# CHECK-INST: lxswu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x65,0xb6,0xd0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

# Scaled indexed loads with a zero-extended 32-bit index (RV64-only).

lxsuwb a0, (a1), a2
# CHECK-INST: lxsuwb a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x05,0xb6,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsuwh a0, (a1), a2
# CHECK-INST: lxsuwh a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x15,0xb6,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsuww a0, (a1), a2
# CHECK-INST: lxsuww a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x25,0xb6,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsuwd a0, (a1), a2
# CHECK-INST: lxsuwd a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x35,0xb6,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsuwbu a0, (a1), a2
# CHECK-INST: lxsuwbu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x45,0xb6,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsuwhu a0, (a1), a2
# CHECK-INST: lxsuwhu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x55,0xb6,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}

lxsuwwu a0, (a1), a2
# CHECK-INST: lxsuwwu a0, (a1), a2
# CHECK-ENCODING: [0x2f,0x65,0xb6,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zilx' (Indexed Integer Load Instructions){{$}}
