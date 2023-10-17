# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

armsll.w $a0, $a1, 1
# CHECK-INST: armsll.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x94,0x3a,0x00]

armsrl.w $a0, $a1, 1
# CHECK-INST: armsrl.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x14,0x3b,0x00]

armsra.w $a0, $a1, 1
# CHECK-INST: armsra.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x94,0x3b,0x00]

armrotr.w $a0, $a1, 1
# CHECK-INST: armrotr.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0x91,0x14,0x3c,0x00]

armslli.w $a0, 1, 1
# CHECK-INST: armslli.w $a0, 1, 1
# CHECK-ENCODING: encoding: [0x91,0x84,0x3c,0x00]

armsrli.w $a0, 1, 1
# CHECK-INST: armsrli.w $a0, 1, 1
# CHECK-ENCODING: encoding: [0x91,0x04,0x3d,0x00]

armsrai.w $a0, 1, 1
# CHECK-INST: armsrai.w $a0, 1, 1
# CHECK-ENCODING: encoding: [0x91,0x84,0x3d,0x00]

armrotri.w $a0, 1, 1
# CHECK-INST: armrotri.w $a0, 1, 1
# CHECK-ENCODING: encoding: [0x91,0x04,0x3e,0x00]

armrrx.w $a0, 1
# CHECK-INST: armrrx.w $a0, 1
# CHECK-ENCODING: encoding: [0x9f,0xc4,0x3f,0x00]
