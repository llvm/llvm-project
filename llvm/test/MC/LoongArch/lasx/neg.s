# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvneg.b $xr23, $xr4
# CHECK-INST: xvneg.b $xr23, $xr4
# CHECK-ENCODING: encoding: [0x97,0x30,0x9c,0x76]

xvneg.h $xr8, $xr14
# CHECK-INST: xvneg.h $xr8, $xr14
# CHECK-ENCODING: encoding: [0xc8,0x35,0x9c,0x76]

xvneg.w $xr23, $xr14
# CHECK-INST: xvneg.w $xr23, $xr14
# CHECK-ENCODING: encoding: [0xd7,0x39,0x9c,0x76]

xvneg.d $xr20, $xr17
# CHECK-INST: xvneg.d $xr20, $xr17
# CHECK-ENCODING: encoding: [0x34,0x3e,0x9c,0x76]
