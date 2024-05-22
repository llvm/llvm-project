# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrln.b.h $xr7, $xr13, $xr5
# CHECK-INST: xvsrln.b.h $xr7, $xr13, $xr5
# CHECK-ENCODING: encoding: [0xa7,0x95,0xf4,0x74]

xvsrln.h.w $xr6, $xr18, $xr5
# CHECK-INST: xvsrln.h.w $xr6, $xr18, $xr5
# CHECK-ENCODING: encoding: [0x46,0x16,0xf5,0x74]

xvsrln.w.d $xr12, $xr12, $xr28
# CHECK-INST: xvsrln.w.d $xr12, $xr12, $xr28
# CHECK-ENCODING: encoding: [0x8c,0xf1,0xf5,0x74]
