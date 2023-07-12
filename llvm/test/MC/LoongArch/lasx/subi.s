# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsubi.bu $xr18, $xr27, 1
# CHECK-INST: xvsubi.bu $xr18, $xr27, 1
# CHECK-ENCODING: encoding: [0x72,0x07,0x8c,0x76]

xvsubi.hu $xr6, $xr23, 19
# CHECK-INST: xvsubi.hu $xr6, $xr23, 19
# CHECK-ENCODING: encoding: [0xe6,0xce,0x8c,0x76]

xvsubi.wu $xr13, $xr3, 5
# CHECK-INST: xvsubi.wu $xr13, $xr3, 5
# CHECK-ENCODING: encoding: [0x6d,0x14,0x8d,0x76]

xvsubi.du $xr26, $xr28, 14
# CHECK-INST: xvsubi.du $xr26, $xr28, 14
# CHECK-ENCODING: encoding: [0x9a,0xbb,0x8d,0x76]
