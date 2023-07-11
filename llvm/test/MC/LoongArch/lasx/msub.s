# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmsub.b $xr22, $xr20, $xr7
# CHECK-INST: xvmsub.b $xr22, $xr20, $xr7
# CHECK-ENCODING: encoding: [0x96,0x1e,0xaa,0x74]

xvmsub.h $xr0, $xr18, $xr12
# CHECK-INST: xvmsub.h $xr0, $xr18, $xr12
# CHECK-ENCODING: encoding: [0x40,0xb2,0xaa,0x74]

xvmsub.w $xr3, $xr22, $xr29
# CHECK-INST: xvmsub.w $xr3, $xr22, $xr29
# CHECK-ENCODING: encoding: [0xc3,0x76,0xab,0x74]

xvmsub.d $xr11, $xr26, $xr2
# CHECK-INST: xvmsub.d $xr11, $xr26, $xr2
# CHECK-ENCODING: encoding: [0x4b,0x8b,0xab,0x74]
