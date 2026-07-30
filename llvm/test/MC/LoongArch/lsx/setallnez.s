# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsetallnez.b $fcc2, $vr8
# CHECK-INST: vsetallnez.b $fcc2, $vr8
# CHECK-ENCODING: encoding: [0x02,0xb1,0x9c,0x72]

vsetallnez.h $fcc0, $vr26
# CHECK-INST: vsetallnez.h $fcc0, $vr26
# CHECK-ENCODING: encoding: [0x40,0xb7,0x9c,0x72]

vsetallnez.w $fcc6, $vr17
# CHECK-INST: vsetallnez.w $fcc6, $vr17
# CHECK-ENCODING: encoding: [0x26,0xba,0x9c,0x72]

vsetallnez.d $fcc0, $vr27
# CHECK-INST: vsetallnez.d $fcc0, $vr27
# CHECK-ENCODING: encoding: [0x60,0xbf,0x9c,0x72]
