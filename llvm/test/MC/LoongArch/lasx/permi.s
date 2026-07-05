# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvpermi.w $xr7, $xr12, 101
# CHECK-INST: xvpermi.w $xr7, $xr12, 101
# CHECK-ENCODING: encoding: [0x87,0x95,0xe5,0x77]

xvpermi.d $xr17, $xr6, 131
# CHECK-INST: xvpermi.d $xr17, $xr6, 131
# CHECK-ENCODING: encoding: [0xd1,0x0c,0xea,0x77]

xvpermi.q $xr10, $xr15, 184
# CHECK-INST: xvpermi.q $xr10, $xr15, 184
# CHECK-ENCODING: encoding: [0xea,0xe1,0xee,0x77]
