# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvpickev.b $xr22, $xr27, $xr6
# CHECK-INST: xvpickev.b $xr22, $xr27, $xr6
# CHECK-ENCODING: encoding: [0x76,0x1b,0x1e,0x75]

xvpickev.h $xr14, $xr11, $xr3
# CHECK-INST: xvpickev.h $xr14, $xr11, $xr3
# CHECK-ENCODING: encoding: [0x6e,0x8d,0x1e,0x75]

xvpickev.w $xr30, $xr28, $xr13
# CHECK-INST: xvpickev.w $xr30, $xr28, $xr13
# CHECK-ENCODING: encoding: [0x9e,0x37,0x1f,0x75]

xvpickev.d $xr1, $xr24, $xr9
# CHECK-INST: xvpickev.d $xr1, $xr24, $xr9
# CHECK-ENCODING: encoding: [0x01,0xa7,0x1f,0x75]

xvpickod.b $xr14, $xr22, $xr15
# CHECK-INST: xvpickod.b $xr14, $xr22, $xr15
# CHECK-ENCODING: encoding: [0xce,0x3e,0x20,0x75]

xvpickod.h $xr31, $xr21, $xr12
# CHECK-INST: xvpickod.h $xr31, $xr21, $xr12
# CHECK-ENCODING: encoding: [0xbf,0xb2,0x20,0x75]

xvpickod.w $xr31, $xr0, $xr30
# CHECK-INST: xvpickod.w $xr31, $xr0, $xr30
# CHECK-ENCODING: encoding: [0x1f,0x78,0x21,0x75]

xvpickod.d $xr10, $xr5, $xr16
# CHECK-INST: xvpickod.d $xr10, $xr5, $xr16
# CHECK-ENCODING: encoding: [0xaa,0xc0,0x21,0x75]
