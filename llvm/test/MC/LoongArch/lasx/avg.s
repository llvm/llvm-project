# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvavg.b $xr5, $xr30, $xr21
# CHECK-INST: xvavg.b $xr5, $xr30, $xr21
# CHECK-ENCODING: encoding: [0xc5,0x57,0x64,0x74]

xvavg.h $xr18, $xr17, $xr21
# CHECK-INST: xvavg.h $xr18, $xr17, $xr21
# CHECK-ENCODING: encoding: [0x32,0xd6,0x64,0x74]

xvavg.w $xr3, $xr23, $xr20
# CHECK-INST: xvavg.w $xr3, $xr23, $xr20
# CHECK-ENCODING: encoding: [0xe3,0x52,0x65,0x74]

xvavg.d $xr27, $xr0, $xr27
# CHECK-INST: xvavg.d $xr27, $xr0, $xr27
# CHECK-ENCODING: encoding: [0x1b,0xec,0x65,0x74]

xvavg.bu $xr11, $xr4, $xr16
# CHECK-INST: xvavg.bu $xr11, $xr4, $xr16
# CHECK-ENCODING: encoding: [0x8b,0x40,0x66,0x74]

xvavg.hu $xr2, $xr1, $xr19
# CHECK-INST: xvavg.hu $xr2, $xr1, $xr19
# CHECK-ENCODING: encoding: [0x22,0xcc,0x66,0x74]

xvavg.wu $xr27, $xr20, $xr27
# CHECK-INST: xvavg.wu $xr27, $xr20, $xr27
# CHECK-ENCODING: encoding: [0x9b,0x6e,0x67,0x74]

xvavg.du $xr23, $xr20, $xr29
# CHECK-INST: xvavg.du $xr23, $xr20, $xr29
# CHECK-ENCODING: encoding: [0x97,0xf6,0x67,0x74]
