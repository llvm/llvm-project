# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

x86adc.b $a0, $a1
# CHECK-INST: x86adc.b $a0, $a1
# CHECK-ENCODING: encoding: [0x8c,0x14,0x3f,0x00]

x86adc.h $a0, $a1
# CHECK-INST: x86adc.h $a0, $a1
# CHECK-ENCODING: encoding: [0x8d,0x14,0x3f,0x00]

x86adc.w $a0, $a1
# CHECK-INST: x86adc.w $a0, $a1
# CHECK-ENCODING: encoding: [0x8e,0x14,0x3f,0x00]

x86adc.d $a0, $a1
# CHECK-INST: x86adc.d $a0, $a1
# CHECK-ENCODING: encoding: [0x8f,0x14,0x3f,0x00]

x86add.b $a0, $a1
# CHECK-INST: x86add.b $a0, $a1
# CHECK-ENCODING: encoding: [0x84,0x14,0x3f,0x00]

x86add.h $a0, $a1
# CHECK-INST: x86add.h $a0, $a1
# CHECK-ENCODING: encoding: [0x85,0x14,0x3f,0x00]

x86add.w $a0, $a1
# CHECK-INST: x86add.w $a0, $a1
# CHECK-ENCODING: encoding: [0x86,0x14,0x3f,0x00]

x86add.d $a0, $a1
# CHECK-INST: x86add.d $a0, $a1
# CHECK-ENCODING: encoding: [0x87,0x14,0x3f,0x00]

x86add.wu $a0, $a1
# CHECK-INST: x86add.wu $a0, $a1
# CHECK-ENCODING: encoding: [0x80,0x14,0x3f,0x00]

x86add.du $a0, $a1
# CHECK-INST: x86add.du $a0, $a1
# CHECK-ENCODING: encoding: [0x81,0x14,0x3f,0x00]

x86inc.b $a0
# CHECK-INST: x86inc.b $a0
# CHECK-ENCODING: encoding: [0x80,0x80,0x00,0x00]

x86inc.h $a0
# CHECK-INST: x86inc.h $a0
# CHECK-ENCODING: encoding: [0x81,0x80,0x00,0x00]

x86inc.w $a0
# CHECK-INST: x86inc.w $a0
# CHECK-ENCODING: encoding: [0x82,0x80,0x00,0x00]

x86inc.d $a0
# CHECK-INST: x86inc.d $a0
# CHECK-ENCODING: encoding: [0x83,0x80,0x00,0x00]

x86sbc.b $a0, $a1
# CHECK-INST: x86sbc.b $a0, $a1
# CHECK-ENCODING: encoding: [0x90,0x14,0x3f,0x00]

x86sbc.h $a0, $a1
# CHECK-INST: x86sbc.h $a0, $a1
# CHECK-ENCODING: encoding: [0x91,0x14,0x3f,0x00]

x86sbc.w $a0, $a1
# CHECK-INST: x86sbc.w $a0, $a1
# CHECK-ENCODING: encoding: [0x92,0x14,0x3f,0x00]

x86sbc.d $a0, $a1
# CHECK-INST: x86sbc.d $a0, $a1
# CHECK-ENCODING: encoding: [0x93,0x14,0x3f,0x00]

x86sub.b $a0, $a1
# CHECK-INST: x86sub.b $a0, $a1
# CHECK-ENCODING: encoding: [0x88,0x14,0x3f,0x00]

x86sub.h $a0, $a1
# CHECK-INST: x86sub.h $a0, $a1
# CHECK-ENCODING: encoding: [0x89,0x14,0x3f,0x00]

x86sub.w $a0, $a1
# CHECK-INST: x86sub.w $a0, $a1
# CHECK-ENCODING: encoding: [0x8a,0x14,0x3f,0x00]

x86sub.d $a0, $a1
# CHECK-INST: x86sub.d $a0, $a1
# CHECK-ENCODING: encoding: [0x8b,0x14,0x3f,0x00]

x86sub.wu $a0, $a1
# CHECK-INST: x86sub.wu $a0, $a1
# CHECK-ENCODING: encoding: [0x82,0x14,0x3f,0x00]

x86sub.du $a0, $a1
# CHECK-INST: x86sub.du $a0, $a1
# CHECK-ENCODING: encoding: [0x83,0x14,0x3f,0x00]

x86dec.b $a0
# CHECK-INST: x86dec.b $a0
# CHECK-ENCODING: encoding: [0x84,0x80,0x00,0x00]

x86dec.h $a0
# CHECK-INST: x86dec.h $a0
# CHECK-ENCODING: encoding: [0x85,0x80,0x00,0x00]

x86dec.w $a0
# CHECK-INST: x86dec.w $a0
# CHECK-ENCODING: encoding: [0x86,0x80,0x00,0x00]

x86dec.d $a0
# CHECK-INST: x86dec.d $a0
# CHECK-ENCODING: encoding: [0x87,0x80,0x00,0x00]

x86and.b $a0, $a1
# CHECK-INST: x86and.b $a0, $a1
# CHECK-ENCODING: encoding: [0x90,0x94,0x3f,0x00]

x86and.h $a0, $a1
# CHECK-INST: x86and.h $a0, $a1
# CHECK-ENCODING: encoding: [0x91,0x94,0x3f,0x00]

x86and.w $a0, $a1
# CHECK-INST: x86and.w $a0, $a1
# CHECK-ENCODING: encoding: [0x92,0x94,0x3f,0x00]

x86and.d $a0, $a1
# CHECK-INST: x86and.d $a0, $a1
# CHECK-ENCODING: encoding: [0x93,0x94,0x3f,0x00]

x86or.b $a0, $a1
# CHECK-INST: x86or.b $a0, $a1
# CHECK-ENCODING: encoding: [0x94,0x94,0x3f,0x00]

x86or.h $a0, $a1
# CHECK-INST: x86or.h $a0, $a1
# CHECK-ENCODING: encoding: [0x95,0x94,0x3f,0x00]

x86or.w $a0, $a1
# CHECK-INST: x86or.w $a0, $a1
# CHECK-ENCODING: encoding: [0x96,0x94,0x3f,0x00]

x86or.d $a0, $a1
# CHECK-INST: x86or.d $a0, $a1
# CHECK-ENCODING: encoding: [0x97,0x94,0x3f,0x00]

x86xor.b $a0, $a1
# CHECK-INST: x86xor.b $a0, $a1
# CHECK-ENCODING: encoding: [0x98,0x94,0x3f,0x00]

x86xor.h $a0, $a1
# CHECK-INST: x86xor.h $a0, $a1
# CHECK-ENCODING: encoding: [0x99,0x94,0x3f,0x00]

x86xor.w $a0, $a1
# CHECK-INST: x86xor.w $a0, $a1
# CHECK-ENCODING: encoding: [0x9a,0x94,0x3f,0x00]

x86xor.d $a0, $a1
# CHECK-INST: x86xor.d $a0, $a1
# CHECK-ENCODING: encoding: [0x9b,0x94,0x3f,0x00]

x86mul.b $a0, $a1
# CHECK-INST: x86mul.b $a0, $a1
# CHECK-ENCODING: encoding: [0x80,0x94,0x3e,0x00]

x86mul.h $a0, $a1
# CHECK-INST: x86mul.h $a0, $a1
# CHECK-ENCODING: encoding: [0x81,0x94,0x3e,0x00]

x86mul.w $a0, $a1
# CHECK-INST: x86mul.w $a0, $a1
# CHECK-ENCODING: encoding: [0x82,0x94,0x3e,0x00]

x86mul.d $a0, $a1
# CHECK-INST: x86mul.d $a0, $a1
# CHECK-ENCODING: encoding: [0x83,0x94,0x3e,0x00]

x86mul.bu $a0, $a1
# CHECK-INST: x86mul.bu $a0, $a1
# CHECK-ENCODING: encoding: [0x84,0x94,0x3e,0x00]

x86mul.hu $a0, $a1
# CHECK-INST: x86mul.hu $a0, $a1
# CHECK-ENCODING: encoding: [0x85,0x94,0x3e,0x00]

x86mul.wu $a0, $a1
# CHECK-INST: x86mul.wu $a0, $a1
# CHECK-ENCODING: encoding: [0x86,0x94,0x3e,0x00]

x86mul.du $a0, $a1
# CHECK-INST: x86mul.du $a0, $a1
# CHECK-ENCODING: encoding: [0x87,0x94,0x3e,0x00]
