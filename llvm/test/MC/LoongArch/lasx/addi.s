# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvaddi.bu $xr1, $xr22, 2
# CHECK-INST: xvaddi.bu $xr1, $xr22, 2
# CHECK-ENCODING: encoding: [0xc1,0x0a,0x8a,0x76]

xvaddi.hu $xr3, $xr10, 29
# CHECK-INST: xvaddi.hu $xr3, $xr10, 29
# CHECK-ENCODING: encoding: [0x43,0xf5,0x8a,0x76]

xvaddi.wu $xr5, $xr11, 3
# CHECK-INST: xvaddi.wu $xr5, $xr11, 3
# CHECK-ENCODING: encoding: [0x65,0x0d,0x8b,0x76]

xvaddi.du $xr6, $xr0, 7
# CHECK-INST: xvaddi.du $xr6, $xr0, 7
# CHECK-ENCODING: encoding: [0x06,0x9c,0x8b,0x76]
