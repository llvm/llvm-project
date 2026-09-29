# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrarn.b.h $xr18, $xr20, $xr15
# CHECK-INST: xvsrarn.b.h $xr18, $xr20, $xr15
# CHECK-ENCODING: encoding: [0x92,0xbe,0xfa,0x74]

xvsrarn.h.w $xr12, $xr1, $xr4
# CHECK-INST: xvsrarn.h.w $xr12, $xr1, $xr4
# CHECK-ENCODING: encoding: [0x2c,0x10,0xfb,0x74]

xvsrarn.w.d $xr9, $xr18, $xr26
# CHECK-INST: xvsrarn.w.d $xr9, $xr18, $xr26
# CHECK-ENCODING: encoding: [0x49,0xea,0xfb,0x74]
