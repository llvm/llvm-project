# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvbitrev.b $xr16, $xr20, $xr3
# CHECK-INST: xvbitrev.b $xr16, $xr20, $xr3
# CHECK-ENCODING: encoding: [0x90,0x0e,0x10,0x75]

xvbitrev.h $xr16, $xr3, $xr20
# CHECK-INST: xvbitrev.h $xr16, $xr3, $xr20
# CHECK-ENCODING: encoding: [0x70,0xd0,0x10,0x75]

xvbitrev.w $xr24, $xr26, $xr23
# CHECK-INST: xvbitrev.w $xr24, $xr26, $xr23
# CHECK-ENCODING: encoding: [0x58,0x5f,0x11,0x75]

xvbitrev.d $xr13, $xr1, $xr27
# CHECK-INST: xvbitrev.d $xr13, $xr1, $xr27
# CHECK-ENCODING: encoding: [0x2d,0xec,0x11,0x75]

xvbitrevi.b $xr7, $xr11, 5
# CHECK-INST: xvbitrevi.b $xr7, $xr11, 5
# CHECK-ENCODING: encoding: [0x67,0x35,0x18,0x77]

xvbitrevi.h $xr1, $xr5, 15
# CHECK-INST: xvbitrevi.h $xr1, $xr5, 15
# CHECK-ENCODING: encoding: [0xa1,0x7c,0x18,0x77]

xvbitrevi.w $xr13, $xr21, 18
# CHECK-INST: xvbitrevi.w $xr13, $xr21, 18
# CHECK-ENCODING: encoding: [0xad,0xca,0x18,0x77]

xvbitrevi.d $xr1, $xr3, 9
# CHECK-INST: xvbitrevi.d $xr1, $xr3, 9
# CHECK-ENCODING: encoding: [0x61,0x24,0x19,0x77]
