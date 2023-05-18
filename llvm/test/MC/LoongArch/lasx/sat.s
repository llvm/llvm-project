# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsat.b $xr22, $xr7, 2
# CHECK-INST: xvsat.b $xr22, $xr7, 2
# CHECK-ENCODING: encoding: [0xf6,0x28,0x24,0x77]

xvsat.h $xr3, $xr0, 5
# CHECK-INST: xvsat.h $xr3, $xr0, 5
# CHECK-ENCODING: encoding: [0x03,0x54,0x24,0x77]

xvsat.w $xr9, $xr16, 0
# CHECK-INST: xvsat.w $xr9, $xr16, 0
# CHECK-ENCODING: encoding: [0x09,0x82,0x24,0x77]

xvsat.d $xr3, $xr8, 1
# CHECK-INST: xvsat.d $xr3, $xr8, 1
# CHECK-ENCODING: encoding: [0x03,0x05,0x25,0x77]

xvsat.bu $xr6, $xr6, 4
# CHECK-INST: xvsat.bu $xr6, $xr6, 4
# CHECK-ENCODING: encoding: [0xc6,0x30,0x28,0x77]

xvsat.hu $xr12, $xr25, 12
# CHECK-INST: xvsat.hu $xr12, $xr25, 12
# CHECK-ENCODING: encoding: [0x2c,0x73,0x28,0x77]

xvsat.wu $xr20, $xr1, 3
# CHECK-INST: xvsat.wu $xr20, $xr1, 3
# CHECK-ENCODING: encoding: [0x34,0x8c,0x28,0x77]

xvsat.du $xr5, $xr20, 7
# CHECK-INST: xvsat.du $xr5, $xr20, 7
# CHECK-ENCODING: encoding: [0x85,0x1e,0x29,0x77]
