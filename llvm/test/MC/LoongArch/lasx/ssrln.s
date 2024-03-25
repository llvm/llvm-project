# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssrln.b.h $xr24, $xr4, $xr4
# CHECK-INST: xvssrln.b.h $xr24, $xr4, $xr4
# CHECK-ENCODING: encoding: [0x98,0x90,0xfc,0x74]

xvssrln.h.w $xr5, $xr15, $xr0
# CHECK-INST: xvssrln.h.w $xr5, $xr15, $xr0
# CHECK-ENCODING: encoding: [0xe5,0x01,0xfd,0x74]

xvssrln.w.d $xr0, $xr25, $xr30
# CHECK-INST: xvssrln.w.d $xr0, $xr25, $xr30
# CHECK-ENCODING: encoding: [0x20,0xfb,0xfd,0x74]

xvssrln.bu.h $xr26, $xr9, $xr26
# CHECK-INST: xvssrln.bu.h $xr26, $xr9, $xr26
# CHECK-ENCODING: encoding: [0x3a,0xe9,0x04,0x75]

xvssrln.hu.w $xr7, $xr20, $xr1
# CHECK-INST: xvssrln.hu.w $xr7, $xr20, $xr1
# CHECK-ENCODING: encoding: [0x87,0x06,0x05,0x75]

xvssrln.wu.d $xr15, $xr13, $xr20
# CHECK-INST: xvssrln.wu.d $xr15, $xr13, $xr20
# CHECK-ENCODING: encoding: [0xaf,0xd1,0x05,0x75]
