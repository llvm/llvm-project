# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfsub.s $xr22, $xr0, $xr3
# CHECK-INST: xvfsub.s $xr22, $xr0, $xr3
# CHECK-ENCODING: encoding: [0x16,0x8c,0x32,0x75]

xvfsub.d $xr4, $xr25, $xr15
# CHECK-INST: xvfsub.d $xr4, $xr25, $xr15
# CHECK-ENCODING: encoding: [0x24,0x3f,0x33,0x75]
