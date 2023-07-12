# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsran.b.h $xr30, $xr13, $xr3
# CHECK-INST: xvsran.b.h $xr30, $xr13, $xr3
# CHECK-ENCODING: encoding: [0xbe,0x8d,0xf6,0x74]

xvsran.h.w $xr18, $xr26, $xr4
# CHECK-INST: xvsran.h.w $xr18, $xr26, $xr4
# CHECK-ENCODING: encoding: [0x52,0x13,0xf7,0x74]

xvsran.w.d $xr27, $xr19, $xr21
# CHECK-INST: xvsran.w.d $xr27, $xr19, $xr21
# CHECK-ENCODING: encoding: [0x7b,0xd6,0xf7,0x74]
