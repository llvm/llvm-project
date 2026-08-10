# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-OBJ

vldi $vr26, -3212
# CHECK-INST: vldi $vr26, -3212
# CHECK-ENCODING: encoding: [0x9a,0x6e,0xe2,0x73]
# CHECK-OBJ: vldi $vr26, -3212

vrepli.b $vr26, -512
# CHECK-INST: vrepli.b $vr26, -512
# CHECK-ENCODING: encoding: [0x1a,0x40,0xe0,0x73]
# CHECK-OBJ: vldi $vr26, 512

vrepli.h $vr26, -512
# CHECK-INST: vrepli.h $vr26, -512
# CHECK-ENCODING: encoding: [0x1a,0xc0,0xe0,0x73]
# CHECK-OBJ: vldi $vr26, 1536

vrepli.w $vr26, -512
# CHECK-INST: vrepli.w $vr26, -512
# CHECK-ENCODING: encoding: [0x1a,0x40,0xe1,0x73]
# CHECK-OBJ: vldi $vr26, 2560

vrepli.d $vr26, -512
# CHECK-INST: vrepli.d $vr26, -512
# CHECK-ENCODING: encoding: [0x1a,0xc0,0xe1,0x73]
# CHECK-OBJ: vldi $vr26, 3584
