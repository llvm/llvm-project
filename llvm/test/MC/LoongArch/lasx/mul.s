# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmul.b $xr18, $xr7, $xr27
# CHECK-INST: xvmul.b $xr18, $xr7, $xr27
# CHECK-ENCODING: encoding: [0xf2,0x6c,0x84,0x74]

xvmul.h $xr9, $xr23, $xr18
# CHECK-INST: xvmul.h $xr9, $xr23, $xr18
# CHECK-ENCODING: encoding: [0xe9,0xca,0x84,0x74]

xvmul.w $xr21, $xr8, $xr27
# CHECK-INST: xvmul.w $xr21, $xr8, $xr27
# CHECK-ENCODING: encoding: [0x15,0x6d,0x85,0x74]

xvmul.d $xr0, $xr15, $xr8
# CHECK-INST: xvmul.d $xr0, $xr15, $xr8
# CHECK-ENCODING: encoding: [0xe0,0xa1,0x85,0x74]
