// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj -o %t.obj %s
// RUN: llvm-readobj -S --sd %t.obj | FileCheck %s  --check-prefix=CHECK-OBJ
// RUN: llvm-readelf -s  %t.obj | FileCheck %s --check-prefix=CHECK-ELF

.section sec00, "ax"
.byte 1
.section sec01, "ax"
nop
nop
.section sec02, "ax"
.balign 4
nop
nop
.section sec03, "ax"
.byte 0
.section sec04, "aw"
nop
nop

// CHECK-OBJ: Name: sec00
// CHECK-OBJ-NEXT: Type: SHT_PROGBITS (0x1)
// CHECK-OBJ-NEXT: Flags [ (0x6)
// CHECK-OBJ: AddressAlignment: 4
// CHECK-OBJ: Name: sec01
// CHECK-OBJ-NEXT: Type: SHT_PROGBITS (0x1)
// CHECK-OBJ-NEXT: Flags [ (0x6)
// CHECK-OBJ: AddressAlignment: 4
// CHECK-OBJ: Name: sec02
// CHECK-OBJ-NEXT: Type: SHT_PROGBITS (0x1)
// CHECK-OBJ-NEXT: Flags [ (0x6)
// CHECK-OBJ: Name: sec03
// CHECK-OBJ-NEXT: Type: SHT_PROGBITS (0x1)
// CHECK-OBJ-NEXT: Flags [ (0x6)
// CHECK-OBJ: AddressAlignment: 4
// CHECK-OBJ: Name: sec04
// CHECK-OBJ-NEXT: Type: SHT_PROGBITS (0x1)
// CHECK-OBJ-NEXT: Flags [ (0x3)
// CHECK-OBJ: AddressAlignment: 1

//CHECK-ELF: sec00             PROGBITS        0000000000000000 000040 000001 00  AX  0   0  4
//CHECK-ELF-NEXT: sec01             PROGBITS        0000000000000000 000044 000008 00  AX  0   0  4
//CHECK-ELF-NEXT: sec02             PROGBITS        0000000000000000 00004c 000008 00  AX  0   0  4
//CHECK-ELF-NEXT: sec03             PROGBITS        0000000000000000 000054 000001 00  AX  0   0  4
//CHECK-ELF-NEXT: sec04             PROGBITS        0000000000000000 000055 000008 00  WA  0   0  1
