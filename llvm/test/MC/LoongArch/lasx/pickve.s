# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvpickve.w $xr25, $xr28, 1
# CHECK-INST: xvpickve.w $xr25, $xr28, 1
# CHECK-ENCODING: encoding: [0x99,0xc7,0x03,0x77]

xvpickve.d $xr13, $xr1, 0
# CHECK-INST: xvpickve.d $xr13, $xr1, 0
# CHECK-ENCODING: encoding: [0x2d,0xe0,0x03,0x77]
