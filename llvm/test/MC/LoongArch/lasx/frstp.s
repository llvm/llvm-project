# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfrstp.b $xr23, $xr18, $xr18
# CHECK-INST: xvfrstp.b $xr23, $xr18, $xr18
# CHECK-ENCODING: encoding: [0x57,0x4a,0x2b,0x75]

xvfrstp.h $xr13, $xr30, $xr6
# CHECK-INST: xvfrstp.h $xr13, $xr30, $xr6
# CHECK-ENCODING: encoding: [0xcd,0x9b,0x2b,0x75]

xvfrstpi.b $xr24, $xr28, 31
# CHECK-INST: xvfrstpi.b $xr24, $xr28, 31
# CHECK-ENCODING: encoding: [0x98,0x7f,0x9a,0x76]

xvfrstpi.h $xr22, $xr24, 18
# CHECK-INST: xvfrstpi.h $xr22, $xr24, 18
# CHECK-ENCODING: encoding: [0x16,0xcb,0x9a,0x76]
