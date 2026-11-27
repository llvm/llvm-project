# RUN: llvm-mc %s -triple=riscv64 -mattr=+h -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -mattr=+h -triple riscv64 < %s \
# RUN:     | llvm-objdump --mattr=+h -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# RUN: llvm-mc --verify -triple riscv32 -mattr=+h < %s

# CHECK-INST: hlv.wu a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x15,0x68]
hlv.wu   a0, (a1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-INST: hlv.wu a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x15,0x68]
hlv.wu   a0, 0(a1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-INST: hlv.d a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x05,0x6c]
hlv.d  a0, (a1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-INST: hlv.d a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x05,0x6c]
hlv.d  a0, 0(a1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-INST: hsv.d a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x6e]
hsv.d   a0, (a1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-INST: hsv.d a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x6e]
hsv.d   a0, 0(a1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
