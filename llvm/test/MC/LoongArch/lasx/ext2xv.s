# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vext2xv.h.b $xr30, $xr19
# CHECK-INST: vext2xv.h.b $xr30, $xr19
# CHECK-ENCODING: encoding: [0x7e,0x12,0x9f,0x76]

vext2xv.w.b $xr27, $xr5
# CHECK-INST: vext2xv.w.b $xr27, $xr5
# CHECK-ENCODING: encoding: [0xbb,0x14,0x9f,0x76]

vext2xv.d.b $xr25, $xr25
# CHECK-INST: vext2xv.d.b $xr25, $xr25
# CHECK-ENCODING: encoding: [0x39,0x1b,0x9f,0x76]

vext2xv.w.h $xr20, $xr20
# CHECK-INST: vext2xv.w.h $xr20, $xr20
# CHECK-ENCODING: encoding: [0x94,0x1e,0x9f,0x76]

vext2xv.d.h $xr8, $xr19
# CHECK-INST: vext2xv.d.h $xr8, $xr19
# CHECK-ENCODING: encoding: [0x68,0x22,0x9f,0x76]

vext2xv.d.w $xr4, $xr25
# CHECK-INST: vext2xv.d.w $xr4, $xr25
# CHECK-ENCODING: encoding: [0x24,0x27,0x9f,0x76]

vext2xv.hu.bu $xr25, $xr12
# CHECK-INST: vext2xv.hu.bu $xr25, $xr12
# CHECK-ENCODING: encoding: [0x99,0x29,0x9f,0x76]

vext2xv.wu.bu $xr31, $xr13
# CHECK-INST: vext2xv.wu.bu $xr31, $xr13
# CHECK-ENCODING: encoding: [0xbf,0x2d,0x9f,0x76]

vext2xv.du.bu $xr12, $xr25
# CHECK-INST: vext2xv.du.bu $xr12, $xr25
# CHECK-ENCODING: encoding: [0x2c,0x33,0x9f,0x76]

vext2xv.wu.hu $xr23, $xr12
# CHECK-INST: vext2xv.wu.hu $xr23, $xr12
# CHECK-ENCODING: encoding: [0x97,0x35,0x9f,0x76]

vext2xv.du.hu $xr18, $xr6
# CHECK-INST: vext2xv.du.hu $xr18, $xr6
# CHECK-ENCODING: encoding: [0xd2,0x38,0x9f,0x76]

vext2xv.du.wu $xr10, $xr21
# CHECK-INST: vext2xv.du.wu $xr10, $xr21
# CHECK-ENCODING: encoding: [0xaa,0x3e,0x9f,0x76]
