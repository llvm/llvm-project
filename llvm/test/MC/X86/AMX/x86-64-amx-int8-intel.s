// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: tdpbssd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd tmm6, tmm5, tmm4

// CHECK: tdpbssd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd tmm3, tmm2, tmm1

// CHECK: tdpbsud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud tmm6, tmm5, tmm4

// CHECK: tdpbsud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud tmm3, tmm2, tmm1

// CHECK: tdpbusd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd tmm6, tmm5, tmm4

// CHECK: tdpbusd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd tmm3, tmm2, tmm1

// CHECK: tdpbuud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud tmm6, tmm5, tmm4

// CHECK: tdpbuud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud tmm3, tmm2, tmm1

// CHECK: tdpbssd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd tmm6, tmm5, tmm4

// CHECK: tdpbssd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd tmm3, tmm2, tmm1

// CHECK: tdpbsud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud tmm6, tmm5, tmm4

// CHECK: tdpbsud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud tmm3, tmm2, tmm1

// CHECK: tdpbusd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd tmm6, tmm5, tmm4

// CHECK: tdpbusd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd tmm3, tmm2, tmm1

// CHECK: tdpbuud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud tmm6, tmm5, tmm4

// CHECK: tdpbuud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud tmm3, tmm2, tmm1

// CHECK: tdpbssd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd tmm6, tmm5, tmm4

// CHECK: tdpbssd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd tmm3, tmm2, tmm1

// CHECK: tdpbsud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud tmm6, tmm5, tmm4

// CHECK: tdpbsud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud tmm3, tmm2, tmm1

// CHECK: tdpbusd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd tmm6, tmm5, tmm4

// CHECK: tdpbusd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd tmm3, tmm2, tmm1

// CHECK: tdpbuud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud tmm6, tmm5, tmm4

// CHECK: tdpbuud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud tmm3, tmm2, tmm1

// CHECK: tdpbssd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd tmm6, tmm5, tmm4

// CHECK: tdpbssd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd tmm3, tmm2, tmm1

// CHECK: tdpbsud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud tmm6, tmm5, tmm4

// CHECK: tdpbsud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud tmm3, tmm2, tmm1

// CHECK: tdpbusd tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd tmm6, tmm5, tmm4

// CHECK: tdpbusd tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd tmm3, tmm2, tmm1

// CHECK: tdpbuud tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud tmm6, tmm5, tmm4

// CHECK: tdpbuud tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud tmm3, tmm2, tmm1
