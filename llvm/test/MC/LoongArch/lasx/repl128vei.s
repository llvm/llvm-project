# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvrepl128vei.b $xr10, $xr19, 2
# CHECK-INST: xvrepl128vei.b $xr10, $xr19, 2
# CHECK-ENCODING: encoding: [0x6a,0x8a,0xf7,0x76]

xvrepl128vei.h $xr6, $xr19, 2
# CHECK-INST: xvrepl128vei.h $xr6, $xr19, 2
# CHECK-ENCODING: encoding: [0x66,0xca,0xf7,0x76]

xvrepl128vei.w $xr11, $xr13, 1
# CHECK-INST: xvrepl128vei.w $xr11, $xr13, 1
# CHECK-ENCODING: encoding: [0xab,0xe5,0xf7,0x76]

xvrepl128vei.d $xr31, $xr23, 0
# CHECK-INST: xvrepl128vei.d $xr31, $xr23, 0
# CHECK-ENCODING: encoding: [0xff,0xf2,0xf7,0x76]
