# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvreplve0.b $xr11, $xr20
# CHECK-INST: xvreplve0.b $xr11, $xr20
# CHECK-ENCODING: encoding: [0x8b,0x02,0x07,0x77]

xvreplve0.h $xr13, $xr26
# CHECK-INST: xvreplve0.h $xr13, $xr26
# CHECK-ENCODING: encoding: [0x4d,0x83,0x07,0x77]

xvreplve0.w $xr8, $xr12
# CHECK-INST: xvreplve0.w $xr8, $xr12
# CHECK-ENCODING: encoding: [0x88,0xc1,0x07,0x77]

xvreplve0.d $xr20, $xr4
# CHECK-INST: xvreplve0.d $xr20, $xr4
# CHECK-ENCODING: encoding: [0x94,0xe0,0x07,0x77]

xvreplve0.q $xr17, $xr20
# CHECK-INST: xvreplve0.q $xr17, $xr20
# CHECK-ENCODING: encoding: [0x91,0xf2,0x07,0x77]
