# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vsc %v11, %v13, 23, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0d,0x0b,0x8c,0x17,0x40,0xb1]
vsc %v11, %v13, 23, %s12

# CHECK-INST: vsc.nc %vix, %s12, 63, 0
# CHECK-ENCODING: encoding: [0x0c,0x00,0x00,0xff,0x00,0x3f,0x20,0xb1]
vsc.nc %vix, %s12, 63, 0

# CHECK-INST: vsc.ot %v63, %vix, -64, %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x3f,0xbf,0x40,0xc0,0xb1]
vsc.ot %v63, %vix, -64, %s63

# CHECK-INST: vsc.nc.ot %v12, %v63, %s12, 0, %vm3
# CHECK-ENCODING: encoding: [0x00,0x00,0x3f,0x0c,0x00,0x8c,0x83,0xb1]
vsc.nc.ot %v12, %v63, %s12, 0, %vm3

# CHECK-INST: vscu %v11, %v13, 23, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0d,0x0b,0x8c,0x17,0x40,0xb2]
vscu %v11, %v13, 23, %s12

# CHECK-INST: vscu.nc %vix, %s12, 63, 0
# CHECK-ENCODING: encoding: [0x0c,0x00,0x00,0xff,0x00,0x3f,0x20,0xb2]
vscu.nc %vix, %s12, 63, 0

# CHECK-INST: vscu.ot %v63, %vix, -64, %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x3f,0xbf,0x40,0xc0,0xb2]
vscu.ot %v63, %vix, -64, %s63

# CHECK-INST: vscu.nc.ot %v12, %v63, %s12, 0, %vm3
# CHECK-ENCODING: encoding: [0x00,0x00,0x3f,0x0c,0x00,0x8c,0x83,0xb2]
vscu.nc.ot %v12, %v63, %s12, 0, %vm3

# CHECK-INST: vscl %v11, %v13, 23, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0d,0x0b,0x8c,0x17,0x40,0xb3]
vscl %v11, %v13, 23, %s12

# CHECK-INST: vscl.ot %vix, %s12, 63, 0
# CHECK-ENCODING: encoding: [0x0c,0x00,0x00,0xff,0x00,0x3f,0xe0,0xb3]
vscl.ot %vix, %s12, 63, 0

# CHECK-INST: vscl.nc %v63, %vix, -64, %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x3f,0xbf,0x40,0x00,0xb3]
vscl.nc %v63, %vix, -64, %s63

# CHECK-INST: vscl.nc.ot %v12, %v63, %s12, 0, %vm3
# CHECK-ENCODING: encoding: [0x00,0x00,0x3f,0x0c,0x00,0x8c,0x83,0xb3]
vscl.nc.ot %v12, %v63, %s12, 0, %vm3
