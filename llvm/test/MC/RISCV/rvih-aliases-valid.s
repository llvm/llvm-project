# RUN: llvm-mc %s -triple=riscv32 -mattr=+h \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ALIAS-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+h \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ALIAS-INST %s
# RUN: llvm-mc -filetype=obj -mattr=+h -triple riscv32 < %s \
# RUN:     | llvm-objdump --mattr=+h -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-NOALIAS-INST %s
# RUN: llvm-mc -filetype=obj -mattr=+h -triple riscv64 < %s \
# RUN:     | llvm-objdump --mattr=+h -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-NOALIAS-INST %s

# CHECK-ALIAS-INST: hfence.gvma{{$}}
# CHECK-NOALIAS-INST: hfence.gvma zero, zero
hfence.gvma

# CHECK-ALIAS-INST: hfence.gvma a0{{$}}
# CHECK-NOALIAS-INST: hfence.gvma a0, zero
hfence.gvma a0

# CHECK-ALIAS-INST: hfence.vvma{{$}}
# CHECK-NOALIAS-INST: hfence.vvma zero, zero
hfence.vvma

# CHECK-ALIAS-INST: hfence.vvma a0{{$}}
# CHECK-NOALIAS-INST: hfence.vvma a0, zero
hfence.vvma a0

# CHECK-INST: hlv.b a0, (a1)
hlv.b   a0, 0(a1)

# CHECK-INST: hlv.bu a0, (a1)
hlv.bu  a0, 0(a1)

# CHECK-INST: hlv.h a1, (a2)
hlv.h   a1, 0(a2)

# CHECK-INST: hlv.hu a1, (a1)
hlv.hu  a1, 0(a1)

# CHECK-INST: hlvx.hu a1, (a2)
hlvx.hu a1, 0(a2)

# CHECK-INST: hlv.w a2, (a2)
hlv.w   a2, 0(a2)

# CHECK-INST: hlvx.wu a2, (a3)
hlvx.wu a2, 0(a3)

# CHECK-INST: hsv.b a0, (a1)
hsv.b   a0, 0(a1)

# CHECK-INST: hsv.h a0, (a1)
hsv.h   a0, 0(a1)

# CHECK-INST: hsv.w a0, (a1)
hsv.w   a0, 0(a1)
