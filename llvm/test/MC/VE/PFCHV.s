# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: pfchv 32, 0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x20,0x40,0x80]
pfchv 32, 0

# CHECK-INST: pfchv.nc %s11, 0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8b,0x00,0x80]
pfchv.nc %s11, 0

# CHECK-INST: pfchv -4, %s13
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8d,0x7c,0x40,0x80]
pfchv -4, %s13

# CHECK-INST: pfchv.nc %s10, %s60
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0xbc,0x8a,0x00,0x80]
pfchv.nc %s10, %s60
