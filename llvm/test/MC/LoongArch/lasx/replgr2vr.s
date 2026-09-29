# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvreplgr2vr.b $xr16, $r16
# CHECK-INST: xvreplgr2vr.b $xr16, $t4
# CHECK-ENCODING: encoding: [0x10,0x02,0x9f,0x76]

xvreplgr2vr.h $xr7, $r22
# CHECK-INST: xvreplgr2vr.h $xr7, $fp
# CHECK-ENCODING: encoding: [0xc7,0x06,0x9f,0x76]

xvreplgr2vr.w $xr4, $r15
# CHECK-INST: xvreplgr2vr.w $xr4, $t3
# CHECK-ENCODING: encoding: [0xe4,0x09,0x9f,0x76]

xvreplgr2vr.d $xr16, $r24
# CHECK-INST: xvreplgr2vr.d $xr16, $s1
# CHECK-ENCODING: encoding: [0x10,0x0f,0x9f,0x76]
