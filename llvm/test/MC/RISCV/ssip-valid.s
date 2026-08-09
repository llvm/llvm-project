# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-ssip \
# RUN:     -M no-aliases -show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 \
# RUN:     -mattr=+experimental-ssip < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-ssip -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-ssip \
# RUN:     -M no-aliases -show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 \
# RUN:     -mattr=+experimental-ssip < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-ssip -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s

# CHECK-INST: sipopret
# CHECK-ENC: encoding: [0x73,0x00,0x80,0x10]
sipopret
