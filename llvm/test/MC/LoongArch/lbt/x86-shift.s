# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

x86rcl.b $a0, $a1
# CHECK-INST: x86rcl.b $a0, $a1
# CHECK-ENCODING: encoding: [0x8c,0x94,0x3f,0x00]

x86rcl.h $a0, $a1
# CHECK-INST: x86rcl.h $a0, $a1
# CHECK-ENCODING: encoding: [0x8d,0x94,0x3f,0x00]

x86rcl.w $a0, $a1
# CHECK-INST: x86rcl.w $a0, $a1
# CHECK-ENCODING: encoding: [0x8e,0x94,0x3f,0x00]

x86rcl.d $a0, $a1
# CHECK-INST: x86rcl.d $a0, $a1
# CHECK-ENCODING: encoding: [0x8f,0x94,0x3f,0x00]

x86rcli.b $a0, 1
# CHECK-INST: x86rcli.b $a0, 1
# CHECK-ENCODING: encoding: [0x98,0x24,0x54,0x00]

x86rcli.h $a0, 1
# CHECK-INST: x86rcli.h $a0, 1
# CHECK-ENCODING: encoding: [0x99,0x44,0x54,0x00]

x86rcli.w $a0, 1
# CHECK-INST: x86rcli.w $a0, 1
# CHECK-ENCODING: encoding: [0x9a,0x84,0x54,0x00]

x86rcli.d $a0, 1
# CHECK-INST: x86rcli.d $a0, 1
# CHECK-ENCODING: encoding: [0x9b,0x04,0x55,0x00]

x86rcr.b $a0, $a1
# CHECK-INST: x86rcr.b $a0, $a1
# CHECK-ENCODING: encoding: [0x88,0x94,0x3f,0x00]

x86rcr.h $a0, $a1
# CHECK-INST: x86rcr.h $a0, $a1
# CHECK-ENCODING: encoding: [0x89,0x94,0x3f,0x00]

x86rcr.w $a0, $a1
# CHECK-INST: x86rcr.w $a0, $a1
# CHECK-ENCODING: encoding: [0x8a,0x94,0x3f,0x00]

x86rcr.d $a0, $a1
# CHECK-INST: x86rcr.d $a0, $a1
# CHECK-ENCODING: encoding: [0x8b,0x94,0x3f,0x00]

x86rcri.b $a0, 1
# CHECK-INST: x86rcri.b $a0, 1
# CHECK-ENCODING: encoding: [0x90,0x24,0x54,0x00]

x86rcri.h $a0, 1
# CHECK-INST: x86rcri.h $a0, 1
# CHECK-ENCODING: encoding: [0x91,0x44,0x54,0x00]

x86rcri.w $a0, 1
# CHECK-INST: x86rcri.w $a0, 1
# CHECK-ENCODING: encoding: [0x92,0x84,0x54,0x00]

x86rcri.d $a0, 1
# CHECK-INST: x86rcri.d $a0, 1
# CHECK-ENCODING: encoding: [0x93,0x04,0x55,0x00]

x86rotl.b $a0, $a1
# CHECK-INST: x86rotl.b $a0, $a1
# CHECK-ENCODING: encoding: [0x84,0x94,0x3f,0x00]

x86rotl.h $a0, $a1
# CHECK-INST: x86rotl.h $a0, $a1
# CHECK-ENCODING: encoding: [0x85,0x94,0x3f,0x00]

x86rotl.w $a0, $a1
# CHECK-INST: x86rotl.w $a0, $a1
# CHECK-ENCODING: encoding: [0x86,0x94,0x3f,0x00]

x86rotl.d $a0, $a1
# CHECK-INST: x86rotl.d $a0, $a1
# CHECK-ENCODING: encoding: [0x87,0x94,0x3f,0x00]

x86rotli.b $a0, 1
# CHECK-INST: x86rotli.b $a0, 1
# CHECK-ENCODING: encoding: [0x94,0x24,0x54,0x00]

x86rotli.h $a0, 1
# CHECK-INST: x86rotli.h $a0, 1
# CHECK-ENCODING: encoding: [0x95,0x44,0x54,0x00]

x86rotli.w $a0, 1
# CHECK-INST: x86rotli.w $a0, 1
# CHECK-ENCODING: encoding: [0x96,0x84,0x54,0x00]

x86rotli.d $a0, 1
# CHECK-INST: x86rotli.d $a0, 1
# CHECK-ENCODING: encoding: [0x97,0x04,0x55,0x00]

x86rotr.b $a0, $a1
# CHECK-INST: x86rotr.b $a0, $a1
# CHECK-ENCODING: encoding: [0x80,0x94,0x3f,0x00]

x86rotr.h $a0, $a1
# CHECK-INST: x86rotr.h $a0, $a1
# CHECK-ENCODING: encoding: [0x81,0x94,0x3f,0x00]

x86rotr.d $a0, $a1
# CHECK-INST: x86rotr.d $a0, $a1
# CHECK-ENCODING: encoding: [0x82,0x94,0x3f,0x00]

x86rotr.w $a0, $a1
# CHECK-INST: x86rotr.w $a0, $a1
# CHECK-ENCODING: encoding: [0x83,0x94,0x3f,0x00]

x86rotri.b $a0, 1
# CHECK-INST: x86rotri.b $a0, 1
# CHECK-ENCODING: encoding: [0x8c,0x24,0x54,0x00]

x86rotri.h $a0, 1
# CHECK-INST: x86rotri.h $a0, 1
# CHECK-ENCODING: encoding: [0x8d,0x44,0x54,0x00]

x86rotri.w $a0, 1
# CHECK-INST: x86rotri.w $a0, 1
# CHECK-ENCODING: encoding: [0x8e,0x84,0x54,0x00]

x86rotri.d $a0, 1
# CHECK-INST: x86rotri.d $a0, 1
# CHECK-ENCODING: encoding: [0x8f,0x04,0x55,0x00]

x86sll.b $a0, $a1
# CHECK-INST: x86sll.b $a0, $a1
# CHECK-ENCODING: encoding: [0x94,0x14,0x3f,0x00]

x86sll.h $a0, $a1
# CHECK-INST: x86sll.h $a0, $a1
# CHECK-ENCODING: encoding: [0x95,0x14,0x3f,0x00]

x86sll.w $a0, $a1
# CHECK-INST: x86sll.w $a0, $a1
# CHECK-ENCODING: encoding: [0x96,0x14,0x3f,0x00]

x86sll.d $a0, $a1
# CHECK-INST: x86sll.d $a0, $a1
# CHECK-ENCODING: encoding: [0x97,0x14,0x3f,0x00]

x86slli.b $a0, 1
# CHECK-INST: x86slli.b $a0, 1
# CHECK-ENCODING: encoding: [0x80,0x24,0x54,0x00]

x86slli.h $a0, 1
# CHECK-INST: x86slli.h $a0, 1
# CHECK-ENCODING: encoding: [0x81,0x44,0x54,0x00]

x86slli.w $a0, 1
# CHECK-INST: x86slli.w $a0, 1
# CHECK-ENCODING: encoding: [0x82,0x84,0x54,0x00]

x86slli.d $a0, 1
# CHECK-INST: x86slli.d $a0, 1
# CHECK-ENCODING: encoding: [0x83,0x04,0x55,0x00]

x86srl.b $a0, $a1
# CHECK-INST: x86srl.b $a0, $a1
# CHECK-ENCODING: encoding: [0x98,0x14,0x3f,0x00]

x86srl.h $a0, $a1
# CHECK-INST: x86srl.h $a0, $a1
# CHECK-ENCODING: encoding: [0x99,0x14,0x3f,0x00]

x86srl.w $a0, $a1
# CHECK-INST: x86srl.w $a0, $a1
# CHECK-ENCODING: encoding: [0x9a,0x14,0x3f,0x00]

x86srl.d $a0, $a1
# CHECK-INST: x86srl.d $a0, $a1
# CHECK-ENCODING: encoding: [0x9b,0x14,0x3f,0x00]

x86srli.b $a0, 1
# CHECK-INST: x86srli.b $a0, 1
# CHECK-ENCODING: encoding: [0x84,0x24,0x54,0x00]

x86srli.h $a0, 1
# CHECK-INST: x86srli.h $a0, 1
# CHECK-ENCODING: encoding: [0x85,0x44,0x54,0x00]

x86srli.w $a0, 1
# CHECK-INST: x86srli.w $a0, 1
# CHECK-ENCODING: encoding: [0x86,0x84,0x54,0x00]

x86srli.d $a0, 1
# CHECK-INST: x86srli.d $a0, 1
# CHECK-ENCODING: encoding: [0x87,0x04,0x55,0x00]

x86sra.b $a0, $a1
# CHECK-INST: x86sra.b $a0, $a1
# CHECK-ENCODING: encoding: [0x9c,0x14,0x3f,0x00]

x86sra.h $a0, $a1
# CHECK-INST: x86sra.h $a0, $a1
# CHECK-ENCODING: encoding: [0x9d,0x14,0x3f,0x00]

x86sra.w $a0, $a1
# CHECK-INST: x86sra.w $a0, $a1
# CHECK-ENCODING: encoding: [0x9e,0x14,0x3f,0x00]

x86sra.d $a0, $a1
# CHECK-INST: x86sra.d $a0, $a1
# CHECK-ENCODING: encoding: [0x9f,0x14,0x3f,0x00]

x86srai.b $a0, 1
# CHECK-INST: x86srai.b $a0, 1
# CHECK-ENCODING: encoding: [0x88,0x24,0x54,0x00]

x86srai.h $a0, 1
# CHECK-INST: x86srai.h $a0, 1
# CHECK-ENCODING: encoding: [0x89,0x44,0x54,0x00]

x86srai.w $a0, 1
# CHECK-INST: x86srai.w $a0, 1
# CHECK-ENCODING: encoding: [0x8a,0x84,0x54,0x00]

x86srai.d $a0, 1
# CHECK-INST: x86srai.d $a0, 1
# CHECK-ENCODING: encoding: [0x8b,0x04,0x55,0x00]
