# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

armadd.w $a0, $a1, 1
# CHECK-INST: armadd.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x14,0x37,0x00]

armsub.w $a0, $a1, 1
# CHECK-INST: armsub.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x94,0x37,0x00]

armadc.w $a0, $a1, 1
# CHECK-INST: armadc.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x14,0x38,0x00]

armsbc.w $a0, $a1, 1
# CHECK-INST: armsbc.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x94,0x38,0x00]

armand.w $a0, $a1, 1
# CHECK-INST: armand.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x14,0x39,0x00]

armor.w $a0, $a1, 1
# CHECK-INST: armor.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x94,0x39,0x00]

armxor.w $a0, $a1, 1
# CHECK-INST: armxor.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x14,0x3a,0x00]

armnot.w $a0, 1
# CHECK-INST: armnot.w $a0, 1
# CHECK-ENCODING: encoding: [0x9c,0xc4,0x3f,0x00]
