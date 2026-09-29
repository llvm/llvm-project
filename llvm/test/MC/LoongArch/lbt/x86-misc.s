# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

x86mfflag $a0, 1
# CHECK-INST: x86mfflag $a0, 1
# CHECK-ENCODING: encoding: [0x04,0x04,0x5c,0x00]

x86mtflag $a0, 1
# CHECK-INST: x86mtflag $a0, 1
# CHECK-ENCODING: encoding: [0x24,0x04,0x5c,0x00]

x86mftop $a0
# CHECK-INST: x86mftop $a0
# CHECK-ENCODING: encoding: [0x04,0x74,0x00,0x00]

x86mttop 1
# CHECK-INST: x86mttop 1
# CHECK-ENCODING: encoding: [0x20,0x70,0x00,0x00]

x86inctop
# CHECK-INST: x86inctop
# CHECK-ENCODING: encoding: [0x09,0x80,0x00,0x00]

x86dectop
# CHECK-INST: x86dectop
# CHECK-ENCODING: encoding: [0x29,0x80,0x00,0x00]

x86settm
# CHECK-INST: x86settm
# CHECK-ENCODING: encoding: [0x08,0x80,0x00,0x00]

x86clrtm
# CHECK-INST: x86clrtm
# CHECK-ENCODING: encoding: [0x28,0x80,0x00,0x00]

x86settag $a0, 1, 1
# CHECK-INST: x86settag $a0, 1, 1
# CHECK-ENCODING: encoding: [0x24,0x04,0x58,0x00]
