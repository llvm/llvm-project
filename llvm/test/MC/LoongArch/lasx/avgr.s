# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvavgr.b $xr29, $xr15, $xr7
# CHECK-INST: xvavgr.b $xr29, $xr15, $xr7
# CHECK-ENCODING: encoding: [0xfd,0x1d,0x68,0x74]

xvavgr.h $xr0, $xr26, $xr15
# CHECK-INST: xvavgr.h $xr0, $xr26, $xr15
# CHECK-ENCODING: encoding: [0x40,0xbf,0x68,0x74]

xvavgr.w $xr23, $xr0, $xr0
# CHECK-INST: xvavgr.w $xr23, $xr0, $xr0
# CHECK-ENCODING: encoding: [0x17,0x00,0x69,0x74]

xvavgr.d $xr29, $xr23, $xr0
# CHECK-INST: xvavgr.d $xr29, $xr23, $xr0
# CHECK-ENCODING: encoding: [0xfd,0x82,0x69,0x74]

xvavgr.bu $xr22, $xr2, $xr25
# CHECK-INST: xvavgr.bu $xr22, $xr2, $xr25
# CHECK-ENCODING: encoding: [0x56,0x64,0x6a,0x74]

xvavgr.hu $xr25, $xr10, $xr21
# CHECK-INST: xvavgr.hu $xr25, $xr10, $xr21
# CHECK-ENCODING: encoding: [0x59,0xd5,0x6a,0x74]

xvavgr.wu $xr17, $xr14, $xr3
# CHECK-INST: xvavgr.wu $xr17, $xr14, $xr3
# CHECK-ENCODING: encoding: [0xd1,0x0d,0x6b,0x74]

xvavgr.du $xr2, $xr11, $xr13
# CHECK-INST: xvavgr.du $xr2, $xr11, $xr13
# CHECK-ENCODING: encoding: [0x62,0xb5,0x6b,0x74]
