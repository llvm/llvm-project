// RUN: llvm-mc -triple x86_64-unknown-unknown -show-encoding %s | FileCheck %s
// some AMX instruction must use SIB.

// CHECK: tdpbssd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd %tmm4, %tmm5, %tmm6

// CHECK: tdpbssd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd %tmm1, %tmm2, %tmm3

// CHECK: tdpbsud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud %tmm4, %tmm5, %tmm6

// CHECK: tdpbsud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud %tmm1, %tmm2, %tmm3

// CHECK: tdpbusd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd %tmm4, %tmm5, %tmm6

// CHECK: tdpbusd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd %tmm1, %tmm2, %tmm3

// CHECK: tdpbuud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud %tmm4, %tmm5, %tmm6

// CHECK: tdpbuud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud %tmm1, %tmm2, %tmm3

// CHECK: tdpbssd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd %tmm4, %tmm5, %tmm6

// CHECK: tdpbssd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd %tmm1, %tmm2, %tmm3

// CHECK: tdpbsud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud %tmm4, %tmm5, %tmm6

// CHECK: tdpbsud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud %tmm1, %tmm2, %tmm3

// CHECK: tdpbusd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd %tmm4, %tmm5, %tmm6

// CHECK: tdpbusd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd %tmm1, %tmm2, %tmm3

// CHECK: tdpbuud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud %tmm4, %tmm5, %tmm6

// CHECK: tdpbuud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud %tmm1, %tmm2, %tmm3

// CHECK: tdpbssd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd %tmm4, %tmm5, %tmm6

// CHECK: tdpbssd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd %tmm1, %tmm2, %tmm3

// CHECK: tdpbsud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud %tmm4, %tmm5, %tmm6

// CHECK: tdpbsud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud %tmm1, %tmm2, %tmm3

// CHECK: tdpbusd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd %tmm4, %tmm5, %tmm6

// CHECK: tdpbusd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd %tmm1, %tmm2, %tmm3

// CHECK: tdpbuud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud %tmm4, %tmm5, %tmm6

// CHECK: tdpbuud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud %tmm1, %tmm2, %tmm3

// CHECK: tdpbssd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5b,0x5e,0xf5]
          tdpbssd %tmm4, %tmm5, %tmm6

// CHECK: tdpbssd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x73,0x5e,0xda]
          tdpbssd %tmm1, %tmm2, %tmm3

// CHECK: tdpbsud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5e,0xf5]
          tdpbsud %tmm4, %tmm5, %tmm6

// CHECK: tdpbsud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5e,0xda]
          tdpbsud %tmm1, %tmm2, %tmm3

// CHECK: tdpbusd %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x59,0x5e,0xf5]
          tdpbusd %tmm4, %tmm5, %tmm6

// CHECK: tdpbusd %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x71,0x5e,0xda]
          tdpbusd %tmm1, %tmm2, %tmm3

// CHECK: tdpbuud %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x58,0x5e,0xf5]
          tdpbuud %tmm4, %tmm5, %tmm6

// CHECK: tdpbuud %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x70,0x5e,0xda]
          tdpbuud %tmm1, %tmm2, %tmm3
