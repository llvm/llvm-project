# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vgt %v11, %v13, 23, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0d,0x0b,0x8c,0x17,0x40,0xa1]
vgt %v11, %v13, 23, %s12

# CHECK-INST: vgt.nc %vix, %s12, 63, 0
# CHECK-ENCODING: encoding: [0x0c,0x00,0x00,0xff,0x00,0x3f,0x20,0xa1]
vgt.nc %vix, %s12, 63, 0

# CHECK-INST: vgt %v63, %vix, -64, %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x3f,0xbf,0x40,0x40,0xa1]
vgt %v63, %vix, -64, %s63

# CHECK-INST: vgt.nc %v12, %v63, %s12, 0, %vm3
# CHECK-ENCODING: encoding: [0x00,0x00,0x3f,0x0c,0x00,0x8c,0x03,0xa1]
vgt.nc %v12, %v63, %s12, 0, %vm3

# CHECK-INST: vgtu %v11, %v13, 23, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0d,0x0b,0x8c,0x17,0x40,0xa2]
vgtu %v11, %v13, 23, %s12

# CHECK-INST: vgtu.nc %vix, %s12, 63, 0
# CHECK-ENCODING: encoding: [0x0c,0x00,0x00,0xff,0x00,0x3f,0x20,0xa2]
vgtu.nc %vix, %s12, 63, 0

# CHECK-INST: vgtu %v63, %vix, -64, %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x3f,0xbf,0x40,0x40,0xa2]
vgtu %v63, %vix, -64, %s63

# CHECK-INST: vgtu.nc %v12, %v63, %s12, 0, %vm3
# CHECK-ENCODING: encoding: [0x00,0x00,0x3f,0x0c,0x00,0x8c,0x03,0xa2]
vgtu.nc %v12, %v63, %s12, 0, %vm3

# CHECK-INST: vgtl.sx %v11, %v13, 23, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0d,0x0b,0x8c,0x17,0x40,0xa3]
vgtl.sx %v11, %v13, 23, %s12

# CHECK-INST: vgtl.zx.nc %vix, %s12, 63, 0
# CHECK-ENCODING: encoding: [0x0c,0x00,0x00,0xff,0x00,0x3f,0xa0,0xa3]
vgtl.nc %vix, %s12, 63, 0

# CHECK-INST: vgtl.zx %v63, %vix, -64, %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x3f,0xbf,0x40,0xc0,0xa3]
vgtl.zx %v63, %vix, -64, %s63

# CHECK-INST: vgtl.sx.nc %v12, %v63, %s12, 0, %vm3
# CHECK-ENCODING: encoding: [0x00,0x00,0x3f,0x0c,0x00,0x8c,0x03,0xa3]
vgtl.sx.nc %v12, %v63, %s12, 0, %vm3
