# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# RUN: not llvm-mc -triple riscv32 < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-INST: hlv.wu a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x15,0x68]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
hlv.wu   a0, (a1)

# CHECK-INST: hlv.wu a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x15,0x68]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
hlv.wu   a0, 0(a1)

# CHECK-INST: hlv.d a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x05,0x6c]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
hlv.d  a0, (a1)

# CHECK-INST: hlv.d a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x05,0x6c]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
hlv.d  a0, 0(a1)

# CHECK-INST: hsv.d a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x6e]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
hsv.d   a0, (a1)

# CHECK-INST: hsv.d a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x6e]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
hsv.d   a0, 0(a1)
