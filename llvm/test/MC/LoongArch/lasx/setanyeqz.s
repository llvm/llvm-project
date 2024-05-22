# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsetanyeqz.b $fcc5, $xr8
# CHECK-INST: xvsetanyeqz.b $fcc5, $xr8
# CHECK-ENCODING: encoding: [0x05,0xa1,0x9c,0x76]

xvsetanyeqz.h $fcc5, $xr20
# CHECK-INST: xvsetanyeqz.h $fcc5, $xr20
# CHECK-ENCODING: encoding: [0x85,0xa6,0x9c,0x76]

xvsetanyeqz.w $fcc7, $xr6
# CHECK-INST: xvsetanyeqz.w $fcc7, $xr6
# CHECK-ENCODING: encoding: [0xc7,0xa8,0x9c,0x76]

xvsetanyeqz.d $fcc6, $xr17
# CHECK-INST: xvsetanyeqz.d $fcc6, $xr17
# CHECK-ENCODING: encoding: [0x26,0xae,0x9c,0x76]
