# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssran.b.h $vr26, $vr26, $vr18
# CHECK-INST: vssran.b.h $vr26, $vr26, $vr18
# CHECK-ENCODING: encoding: [0x5a,0xcb,0xfe,0x70]

vssran.h.w $vr21, $vr14, $vr11
# CHECK-INST: vssran.h.w $vr21, $vr14, $vr11
# CHECK-ENCODING: encoding: [0xd5,0x2d,0xff,0x70]

vssran.w.d $vr4, $vr21, $vr11
# CHECK-INST: vssran.w.d $vr4, $vr21, $vr11
# CHECK-ENCODING: encoding: [0xa4,0xae,0xff,0x70]

vssran.bu.h $vr10, $vr30, $vr19
# CHECK-INST: vssran.bu.h $vr10, $vr30, $vr19
# CHECK-ENCODING: encoding: [0xca,0xcf,0x06,0x71]

vssran.hu.w $vr7, $vr8, $vr20
# CHECK-INST: vssran.hu.w $vr7, $vr8, $vr20
# CHECK-ENCODING: encoding: [0x07,0x51,0x07,0x71]

vssran.wu.d $vr10, $vr21, $vr0
# CHECK-INST: vssran.wu.d $vr10, $vr21, $vr0
# CHECK-ENCODING: encoding: [0xaa,0x82,0x07,0x71]
