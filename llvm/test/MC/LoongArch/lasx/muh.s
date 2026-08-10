# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmuh.b $xr4, $xr8, $xr4
# CHECK-INST: xvmuh.b $xr4, $xr8, $xr4
# CHECK-ENCODING: encoding: [0x04,0x11,0x86,0x74]

xvmuh.h $xr5, $xr23, $xr26
# CHECK-INST: xvmuh.h $xr5, $xr23, $xr26
# CHECK-ENCODING: encoding: [0xe5,0xea,0x86,0x74]

xvmuh.w $xr28, $xr3, $xr25
# CHECK-INST: xvmuh.w $xr28, $xr3, $xr25
# CHECK-ENCODING: encoding: [0x7c,0x64,0x87,0x74]

xvmuh.d $xr6, $xr0, $xr9
# CHECK-INST: xvmuh.d $xr6, $xr0, $xr9
# CHECK-ENCODING: encoding: [0x06,0xa4,0x87,0x74]

xvmuh.bu $xr15, $xr20, $xr24
# CHECK-INST: xvmuh.bu $xr15, $xr20, $xr24
# CHECK-ENCODING: encoding: [0x8f,0x62,0x88,0x74]

xvmuh.hu $xr28, $xr12, $xr27
# CHECK-INST: xvmuh.hu $xr28, $xr12, $xr27
# CHECK-ENCODING: encoding: [0x9c,0xed,0x88,0x74]

xvmuh.wu $xr25, $xr6, $xr10
# CHECK-INST: xvmuh.wu $xr25, $xr6, $xr10
# CHECK-ENCODING: encoding: [0xd9,0x28,0x89,0x74]

xvmuh.du $xr19, $xr8, $xr31
# CHECK-INST: xvmuh.du $xr19, $xr8, $xr31
# CHECK-ENCODING: encoding: [0x13,0xfd,0x89,0x74]
