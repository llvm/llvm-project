# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvandi.b $xr11, $xr7, 66
# CHECK-INST: xvandi.b $xr11, $xr7, 66
# CHECK-ENCODING: encoding: [0xeb,0x08,0xd1,0x77]
