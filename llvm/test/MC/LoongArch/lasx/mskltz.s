# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmskltz.b $xr14, $xr5
# CHECK-INST: xvmskltz.b $xr14, $xr5
# CHECK-ENCODING: encoding: [0xae,0x40,0x9c,0x76]

xvmskltz.h $xr11, $xr25
# CHECK-INST: xvmskltz.h $xr11, $xr25
# CHECK-ENCODING: encoding: [0x2b,0x47,0x9c,0x76]

xvmskltz.w $xr14, $xr27
# CHECK-INST: xvmskltz.w $xr14, $xr27
# CHECK-ENCODING: encoding: [0x6e,0x4b,0x9c,0x76]

xvmskltz.d $xr7, $xr23
# CHECK-INST: xvmskltz.d $xr7, $xr23
# CHECK-ENCODING: encoding: [0xe7,0x4e,0x9c,0x76]
