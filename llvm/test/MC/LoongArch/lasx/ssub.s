# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssub.b $xr14, $xr19, $xr24
# CHECK-INST: xvssub.b $xr14, $xr19, $xr24
# CHECK-ENCODING: encoding: [0x6e,0x62,0x48,0x74]

xvssub.h $xr13, $xr8, $xr19
# CHECK-INST: xvssub.h $xr13, $xr8, $xr19
# CHECK-ENCODING: encoding: [0x0d,0xcd,0x48,0x74]

xvssub.w $xr28, $xr27, $xr28
# CHECK-INST: xvssub.w $xr28, $xr27, $xr28
# CHECK-ENCODING: encoding: [0x7c,0x73,0x49,0x74]

xvssub.d $xr28, $xr16, $xr2
# CHECK-INST: xvssub.d $xr28, $xr16, $xr2
# CHECK-ENCODING: encoding: [0x1c,0x8a,0x49,0x74]

xvssub.bu $xr11, $xr13, $xr17
# CHECK-INST: xvssub.bu $xr11, $xr13, $xr17
# CHECK-ENCODING: encoding: [0xab,0x45,0x4c,0x74]

xvssub.hu $xr16, $xr10, $xr28
# CHECK-INST: xvssub.hu $xr16, $xr10, $xr28
# CHECK-ENCODING: encoding: [0x50,0xf1,0x4c,0x74]

xvssub.wu $xr21, $xr0, $xr13
# CHECK-INST: xvssub.wu $xr21, $xr0, $xr13
# CHECK-ENCODING: encoding: [0x15,0x34,0x4d,0x74]

xvssub.du $xr18, $xr26, $xr27
# CHECK-INST: xvssub.du $xr18, $xr26, $xr27
# CHECK-ENCODING: encoding: [0x52,0xef,0x4d,0x74]
