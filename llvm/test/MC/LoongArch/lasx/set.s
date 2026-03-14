# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvseteqz.v $fcc7, $xr1
# CHECK-INST: xvseteqz.v $fcc7, $xr1
# CHECK-ENCODING: encoding: [0x27,0x98,0x9c,0x76]

xvsetnez.v $fcc7, $xr13
# CHECK-INST: xvsetnez.v $fcc7, $xr13
# CHECK-ENCODING: encoding: [0xa7,0x9d,0x9c,0x76]
