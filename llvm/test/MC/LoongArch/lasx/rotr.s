# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvrotr.b $xr0, $xr6, $xr30
# CHECK-INST: xvrotr.b $xr0, $xr6, $xr30
# CHECK-ENCODING: encoding: [0xc0,0x78,0xee,0x74]

xvrotr.h $xr19, $xr17, $xr10
# CHECK-INST: xvrotr.h $xr19, $xr17, $xr10
# CHECK-ENCODING: encoding: [0x33,0xaa,0xee,0x74]

xvrotr.w $xr18, $xr2, $xr7
# CHECK-INST: xvrotr.w $xr18, $xr2, $xr7
# CHECK-ENCODING: encoding: [0x52,0x1c,0xef,0x74]

xvrotr.d $xr11, $xr23, $xr11
# CHECK-INST: xvrotr.d $xr11, $xr23, $xr11
# CHECK-ENCODING: encoding: [0xeb,0xae,0xef,0x74]

xvrotri.b $xr1, $xr5, 3
# CHECK-INST: xvrotri.b $xr1, $xr5, 3
# CHECK-ENCODING: encoding: [0xa1,0x2c,0xa0,0x76]

xvrotri.h $xr1, $xr17, 3
# CHECK-INST: xvrotri.h $xr1, $xr17, 3
# CHECK-ENCODING: encoding: [0x21,0x4e,0xa0,0x76]

xvrotri.w $xr25, $xr23, 19
# CHECK-INST: xvrotri.w $xr25, $xr23, 19
# CHECK-ENCODING: encoding: [0xf9,0xce,0xa0,0x76]

xvrotri.d $xr7, $xr24, 37
# CHECK-INST: xvrotri.d $xr7, $xr24, 37
# CHECK-ENCODING: encoding: [0x07,0x97,0xa1,0x76]
