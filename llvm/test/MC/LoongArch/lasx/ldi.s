# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-OBJ

xvldi $xr31, 3206
# CHECK-INST: xvldi $xr31, 3206
# CHECK-ENCODING: encoding: [0xdf,0x90,0xe1,0x77]
# CHECK-OBJ: vldi $xr31, 3206

xvrepli.b $xr26, -512
# CHECK-INST: vrepli.b $xr26, -512
# CHECK-ENCODING: encoding: [0x1a,0x40,0xe0,0x77]
# CHECK-OBJ: vldi $xr26, 512

xvrepli.h $xr26, -512
# CHECK-INST: vrepli.h $xr26, -512
# CHECK-ENCODING: encoding: [0x1a,0xc0,0xe0,0x77]
# CHECK-OBJ: vldi $xr26, 1536

xvrepli.w $xr26, -512
# CHECK-INST: vrepli.w $xr26, -512
# CHECK-ENCODING: encoding: [0x1a,0x40,0xe1,0x77]
# CHECK-OBJ: vldi $xr26, 2560

xvrepli.d $xr26, -512
# CHECK-INST: vrepli.d $xr26, -512
# CHECK-ENCODING: encoding: [0x1a,0xc0,0xe1,0x77]
# CHECK-OBJ: vldi $xr26, 3584
