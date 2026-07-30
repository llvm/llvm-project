# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmadd.b $xr5, $xr31, $xr8
# CHECK-INST: xvmadd.b $xr5, $xr31, $xr8
# CHECK-ENCODING: encoding: [0xe5,0x23,0xa8,0x74]

xvmadd.h $xr4, $xr0, $xr28
# CHECK-INST: xvmadd.h $xr4, $xr0, $xr28
# CHECK-ENCODING: encoding: [0x04,0xf0,0xa8,0x74]

xvmadd.w $xr2, $xr13, $xr24
# CHECK-INST: xvmadd.w $xr2, $xr13, $xr24
# CHECK-ENCODING: encoding: [0xa2,0x61,0xa9,0x74]

xvmadd.d $xr19, $xr8, $xr18
# CHECK-INST: xvmadd.d $xr19, $xr8, $xr18
# CHECK-ENCODING: encoding: [0x13,0xc9,0xa9,0x74]
