# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vpickve2gr.b $r18, $vr1, 1
# CHECK-INST: vpickve2gr.b $t6, $vr1, 1
# CHECK-ENCODING: encoding: [0x32,0x84,0xef,0x72]

vpickve2gr.h $r2, $vr5, 3
# CHECK-INST: vpickve2gr.h $tp, $vr5, 3
# CHECK-ENCODING: encoding: [0xa2,0xcc,0xef,0x72]

vpickve2gr.w $r3, $vr11, 2
# CHECK-INST: vpickve2gr.w $sp, $vr11, 2
# CHECK-ENCODING: encoding: [0x63,0xe9,0xef,0x72]

vpickve2gr.d $r26, $vr1, 1
# CHECK-INST: vpickve2gr.d $s3, $vr1, 1
# CHECK-ENCODING: encoding: [0x3a,0xf4,0xef,0x72]

vpickve2gr.bu $r28, $vr14, 6
# CHECK-INST: vpickve2gr.bu $s5, $vr14, 6
# CHECK-ENCODING: encoding: [0xdc,0x99,0xf3,0x72]

vpickve2gr.hu $r7, $vr6, 7
# CHECK-INST: vpickve2gr.hu $a3, $vr6, 7
# CHECK-ENCODING: encoding: [0xc7,0xdc,0xf3,0x72]

vpickve2gr.wu $r11, $vr30, 1
# CHECK-INST: vpickve2gr.wu $a7, $vr30, 1
# CHECK-ENCODING: encoding: [0xcb,0xe7,0xf3,0x72]

vpickve2gr.du $r13, $vr5, 0
# CHECK-INST: vpickve2gr.du $t1, $vr5, 0
# CHECK-ENCODING: encoding: [0xad,0xf0,0xf3,0x72]
