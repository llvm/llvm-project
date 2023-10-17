# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvextl.q.d $xr29, $xr12
# CHECK-INST: xvextl.q.d $xr29, $xr12
# CHECK-ENCODING: encoding: [0x9d,0x01,0x09,0x77]

xvextl.qu.du $xr27, $xr20
# CHECK-INST: xvextl.qu.du $xr27, $xr20
# CHECK-ENCODING: encoding: [0x9b,0x02,0x0d,0x77]
