# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvclz.b $xr5, $xr6
# CHECK-INST: xvclz.b $xr5, $xr6
# CHECK-ENCODING: encoding: [0xc5,0x10,0x9c,0x76]

xvclz.h $xr4, $xr7
# CHECK-INST: xvclz.h $xr4, $xr7
# CHECK-ENCODING: encoding: [0xe4,0x14,0x9c,0x76]

xvclz.w $xr12, $xr0
# CHECK-INST: xvclz.w $xr12, $xr0
# CHECK-ENCODING: encoding: [0x0c,0x18,0x9c,0x76]

xvclz.d $xr1, $xr0
# CHECK-INST: xvclz.d $xr1, $xr0
# CHECK-ENCODING: encoding: [0x01,0x1c,0x9c,0x76]
