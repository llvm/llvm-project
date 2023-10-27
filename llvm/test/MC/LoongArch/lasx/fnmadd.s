# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfnmadd.s $xr14, $xr22, $xr23, $xr24
# CHECK-INST: xvfnmadd.s $xr14, $xr22, $xr23, $xr24
# CHECK-ENCODING: encoding: [0xce,0x5e,0x9c,0x0a]

xvfnmadd.d $xr1, $xr30, $xr23, $xr12
# CHECK-INST: xvfnmadd.d $xr1, $xr30, $xr23, $xr12
# CHECK-ENCODING: encoding: [0xc1,0x5f,0xa6,0x0a]
