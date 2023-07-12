# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvpickve2gr.w $r14, $xr11, 6
# CHECK-INST: xvpickve2gr.w $t2, $xr11, 6
# CHECK-ENCODING: encoding: [0x6e,0xd9,0xef,0x76]

xvpickve2gr.d $r8, $xr6, 0
# CHECK-INST: xvpickve2gr.d $a4, $xr6, 0
# CHECK-ENCODING: encoding: [0xc8,0xe0,0xef,0x76]

xvpickve2gr.wu $r12, $xr1, 4
# CHECK-INST: xvpickve2gr.wu $t0, $xr1, 4
# CHECK-ENCODING: encoding: [0x2c,0xd0,0xf3,0x76]

xvpickve2gr.du $r10, $xr8, 0
# CHECK-INST: xvpickve2gr.du $a6, $xr8, 0
# CHECK-ENCODING: encoding: [0x0a,0xe1,0xf3,0x76]
