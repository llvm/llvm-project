# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvbitset.b $xr6, $xr16, $xr28
# CHECK-INST: xvbitset.b $xr6, $xr16, $xr28
# CHECK-ENCODING: encoding: [0x06,0x72,0x0e,0x75]

xvbitset.h $xr5, $xr13, $xr31
# CHECK-INST: xvbitset.h $xr5, $xr13, $xr31
# CHECK-ENCODING: encoding: [0xa5,0xfd,0x0e,0x75]

xvbitset.w $xr7, $xr28, $xr8
# CHECK-INST: xvbitset.w $xr7, $xr28, $xr8
# CHECK-ENCODING: encoding: [0x87,0x23,0x0f,0x75]

xvbitset.d $xr4, $xr16, $xr12
# CHECK-INST: xvbitset.d $xr4, $xr16, $xr12
# CHECK-ENCODING: encoding: [0x04,0xb2,0x0f,0x75]

xvbitseti.b $xr26, $xr3, 0
# CHECK-INST: xvbitseti.b $xr26, $xr3, 0
# CHECK-ENCODING: encoding: [0x7a,0x20,0x14,0x77]

xvbitseti.h $xr9, $xr19, 9
# CHECK-INST: xvbitseti.h $xr9, $xr19, 9
# CHECK-ENCODING: encoding: [0x69,0x66,0x14,0x77]

xvbitseti.w $xr12, $xr19, 2
# CHECK-INST: xvbitseti.w $xr12, $xr19, 2
# CHECK-ENCODING: encoding: [0x6c,0x8a,0x14,0x77]

xvbitseti.d $xr20, $xr7, 2
# CHECK-INST: xvbitseti.d $xr20, $xr7, 2
# CHECK-ENCODING: encoding: [0xf4,0x08,0x15,0x77]
