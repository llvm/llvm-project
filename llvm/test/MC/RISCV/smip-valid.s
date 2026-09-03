# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-smip \
# RUN:     -M no-aliases -show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 \
# RUN:     -mattr=+experimental-smip < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-smip -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-smip \
# RUN:     -M no-aliases -show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 \
# RUN:     -mattr=+experimental-smip < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-smip -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s

# CHECK-INST: mipopret
# CHECK-ENC: encoding: [0x73,0x00,0x80,0x30]
mipopret
