# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvseq.b $xr3, $xr4, $xr19
# CHECK-INST: xvseq.b $xr3, $xr4, $xr19
# CHECK-ENCODING: encoding: [0x83,0x4c,0x00,0x74]

xvseq.h $xr0, $xr21, $xr5
# CHECK-INST: xvseq.h $xr0, $xr21, $xr5
# CHECK-ENCODING: encoding: [0xa0,0x96,0x00,0x74]

xvseq.w $xr6, $xr16, $xr19
# CHECK-INST: xvseq.w $xr6, $xr16, $xr19
# CHECK-ENCODING: encoding: [0x06,0x4e,0x01,0x74]

xvseq.d $xr8, $xr13, $xr13
# CHECK-INST: xvseq.d $xr8, $xr13, $xr13
# CHECK-ENCODING: encoding: [0xa8,0xb5,0x01,0x74]

xvseqi.b $xr12, $xr25, 0
# CHECK-INST: xvseqi.b $xr12, $xr25, 0
# CHECK-ENCODING: encoding: [0x2c,0x03,0x80,0x76]

xvseqi.h $xr9, $xr4, 10
# CHECK-INST: xvseqi.h $xr9, $xr4, 10
# CHECK-ENCODING: encoding: [0x89,0xa8,0x80,0x76]

xvseqi.w $xr25, $xr4, -12
# CHECK-INST: xvseqi.w $xr25, $xr4, -12
# CHECK-ENCODING: encoding: [0x99,0x50,0x81,0x76]

xvseqi.d $xr11, $xr7, 7
# CHECK-INST: xvseqi.d $xr11, $xr7, 7
# CHECK-ENCODING: encoding: [0xeb,0x9c,0x81,0x76]
