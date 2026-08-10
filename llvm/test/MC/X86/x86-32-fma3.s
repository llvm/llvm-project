// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vfmadd132pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x98,0xca]
          vfmadd132pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd132pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x98,0x08]
          vfmadd132pd  (%eax), %xmm5, %xmm1

// CHECK: vfmadd132ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x98,0xca]
          vfmadd132ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd132ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x98,0x08]
          vfmadd132ps  (%eax), %xmm5, %xmm1

// CHECK: vfmadd213pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa8,0xca]
          vfmadd213pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd213pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa8,0x08]
          vfmadd213pd  (%eax), %xmm5, %xmm1

// CHECK: vfmadd213ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa8,0xca]
          vfmadd213ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd213ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa8,0x08]
          vfmadd213ps  (%eax), %xmm5, %xmm1

// CHECK: vfmadd231pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb8,0xca]
          vfmadd231pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd231pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb8,0x08]
          vfmadd231pd  (%eax), %xmm5, %xmm1

// CHECK: vfmadd231ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb8,0xca]
          vfmadd231ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd231ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb8,0x08]
          vfmadd231ps  (%eax), %xmm5, %xmm1

// CHECK: vfmadd132pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x98,0xca]
          vfmadd132pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd132pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x98,0x08]
          vfmadd132pd  (%eax), %ymm5, %ymm1

// CHECK: vfmadd132ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x98,0xca]
          vfmadd132ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd132ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x98,0x08]
          vfmadd132ps  (%eax), %ymm5, %ymm1

// CHECK: vfmadd213pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa8,0xca]
          vfmadd213pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd213pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa8,0x08]
          vfmadd213pd  (%eax), %ymm5, %ymm1

// CHECK: vfmadd213ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa8,0xca]
          vfmadd213ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd213ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa8,0x08]
          vfmadd213ps  (%eax), %ymm5, %ymm1

// CHECK: vfmadd231pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb8,0xca]
          vfmadd231pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd231pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb8,0x08]
          vfmadd231pd  (%eax), %ymm5, %ymm1

// CHECK: vfmadd231ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb8,0xca]
          vfmadd231ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd231ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb8,0x08]
          vfmadd231ps  (%eax), %ymm5, %ymm1

// CHECK: vfmadd132pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x98,0xca]
          vfmadd132pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd132pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x98,0x08]
          vfmadd132pd  (%eax), %xmm5, %xmm1

// CHECK: vfmadd132ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x98,0xca]
          vfmadd132ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd132ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x98,0x08]
          vfmadd132ps  (%eax), %xmm5, %xmm1

// CHECK: vfmadd213pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa8,0xca]
          vfmadd213pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd213pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa8,0x08]
          vfmadd213pd  (%eax), %xmm5, %xmm1

// CHECK: vfmadd213ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa8,0xca]
          vfmadd213ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd213ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa8,0x08]
          vfmadd213ps  (%eax), %xmm5, %xmm1

// CHECK: vfmadd231pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb8,0xca]
          vfmadd231pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd231pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb8,0x08]
          vfmadd231pd  (%eax), %xmm5, %xmm1

// CHECK: vfmadd231ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb8,0xca]
          vfmadd231ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmadd231ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb8,0x08]
          vfmadd231ps  (%eax), %xmm5, %xmm1

// CHECK: vfmaddsub132pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x96,0xca]
          vfmaddsub132pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmaddsub132pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x96,0x08]
          vfmaddsub132pd  (%eax), %xmm5, %xmm1

// CHECK: vfmaddsub132ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x96,0xca]
          vfmaddsub132ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmaddsub132ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x96,0x08]
          vfmaddsub132ps  (%eax), %xmm5, %xmm1

// CHECK: vfmaddsub213pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa6,0xca]
          vfmaddsub213pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmaddsub213pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa6,0x08]
          vfmaddsub213pd  (%eax), %xmm5, %xmm1

// CHECK: vfmaddsub213ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa6,0xca]
          vfmaddsub213ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmaddsub213ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa6,0x08]
          vfmaddsub213ps  (%eax), %xmm5, %xmm1

// CHECK: vfmaddsub231pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb6,0xca]
          vfmaddsub231pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmaddsub231pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb6,0x08]
          vfmaddsub231pd  (%eax), %xmm5, %xmm1

// CHECK: vfmaddsub231ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb6,0xca]
          vfmaddsub231ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmaddsub231ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb6,0x08]
          vfmaddsub231ps  (%eax), %xmm5, %xmm1

// CHECK: vfmsubadd132pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x97,0xca]
          vfmsubadd132pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmsubadd132pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x97,0x08]
          vfmsubadd132pd  (%eax), %xmm5, %xmm1

// CHECK: vfmsubadd132ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x97,0xca]
          vfmsubadd132ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmsubadd132ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x97,0x08]
          vfmsubadd132ps  (%eax), %xmm5, %xmm1

// CHECK: vfmsubadd213pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa7,0xca]
          vfmsubadd213pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmsubadd213pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xa7,0x08]
          vfmsubadd213pd  (%eax), %xmm5, %xmm1

// CHECK: vfmsubadd213ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa7,0xca]
          vfmsubadd213ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmsubadd213ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xa7,0x08]
          vfmsubadd213ps  (%eax), %xmm5, %xmm1

// CHECK: vfmsubadd231pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb7,0xca]
          vfmsubadd231pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmsubadd231pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xb7,0x08]
          vfmsubadd231pd  (%eax), %xmm5, %xmm1

// CHECK: vfmsubadd231ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb7,0xca]
          vfmsubadd231ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmsubadd231ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xb7,0x08]
          vfmsubadd231ps  (%eax), %xmm5, %xmm1

// CHECK: vfmsub132pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x9a,0xca]
          vfmsub132pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmsub132pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x9a,0x08]
          vfmsub132pd  (%eax), %xmm5, %xmm1

// CHECK: vfmsub132ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x9a,0xca]
          vfmsub132ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmsub132ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x9a,0x08]
          vfmsub132ps  (%eax), %xmm5, %xmm1

// CHECK: vfmsub213pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xaa,0xca]
          vfmsub213pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmsub213pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xaa,0x08]
          vfmsub213pd  (%eax), %xmm5, %xmm1

// CHECK: vfmsub213ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xaa,0xca]
          vfmsub213ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmsub213ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xaa,0x08]
          vfmsub213ps  (%eax), %xmm5, %xmm1

// CHECK: vfmsub231pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xba,0xca]
          vfmsub231pd  %xmm2, %xmm5, %xmm1

// CHECK: vfmsub231pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xba,0x08]
          vfmsub231pd  (%eax), %xmm5, %xmm1

// CHECK: vfmsub231ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xba,0xca]
          vfmsub231ps  %xmm2, %xmm5, %xmm1

// CHECK: vfmsub231ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xba,0x08]
          vfmsub231ps  (%eax), %xmm5, %xmm1

// CHECK: vfnmadd132pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x9c,0xca]
          vfnmadd132pd  %xmm2, %xmm5, %xmm1

// CHECK: vfnmadd132pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x9c,0x08]
          vfnmadd132pd  (%eax), %xmm5, %xmm1

// CHECK: vfnmadd132ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x9c,0xca]
          vfnmadd132ps  %xmm2, %xmm5, %xmm1

// CHECK: vfnmadd132ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x9c,0x08]
          vfnmadd132ps  (%eax), %xmm5, %xmm1

// CHECK: vfnmadd213pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xac,0xca]
          vfnmadd213pd  %xmm2, %xmm5, %xmm1

// CHECK: vfnmadd213pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xac,0x08]
          vfnmadd213pd  (%eax), %xmm5, %xmm1

// CHECK: vfnmadd213ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xac,0xca]
          vfnmadd213ps  %xmm2, %xmm5, %xmm1

// CHECK: vfnmadd213ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xac,0x08]
          vfnmadd213ps  (%eax), %xmm5, %xmm1

// CHECK: vfnmadd231pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xbc,0xca]
          vfnmadd231pd  %xmm2, %xmm5, %xmm1

// CHECK: vfnmadd231pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xbc,0x08]
          vfnmadd231pd  (%eax), %xmm5, %xmm1

// CHECK: vfnmadd231ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xbc,0xca]
          vfnmadd231ps  %xmm2, %xmm5, %xmm1

// CHECK: vfnmadd231ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xbc,0x08]
          vfnmadd231ps  (%eax), %xmm5, %xmm1

// CHECK: vfnmsub132pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x9e,0xca]
          vfnmsub132pd  %xmm2, %xmm5, %xmm1

// CHECK: vfnmsub132pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0x9e,0x08]
          vfnmsub132pd  (%eax), %xmm5, %xmm1

// CHECK: vfnmsub132ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x9e,0xca]
          vfnmsub132ps  %xmm2, %xmm5, %xmm1

// CHECK: vfnmsub132ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x9e,0x08]
          vfnmsub132ps  (%eax), %xmm5, %xmm1

// CHECK: vfnmsub213pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xae,0xca]
          vfnmsub213pd  %xmm2, %xmm5, %xmm1

// CHECK: vfnmsub213pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xae,0x08]
          vfnmsub213pd  (%eax), %xmm5, %xmm1

// CHECK: vfnmsub213ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xae,0xca]
          vfnmsub213ps  %xmm2, %xmm5, %xmm1

// CHECK: vfnmsub213ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xae,0x08]
          vfnmsub213ps  (%eax), %xmm5, %xmm1

// CHECK: vfnmsub231pd  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xbe,0xca]
          vfnmsub231pd  %xmm2, %xmm5, %xmm1

// CHECK: vfnmsub231pd  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xd1,0xbe,0x08]
          vfnmsub231pd  (%eax), %xmm5, %xmm1

// CHECK: vfnmsub231ps  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xbe,0xca]
          vfnmsub231ps  %xmm2, %xmm5, %xmm1

// CHECK: vfnmsub231ps  (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xbe,0x08]
          vfnmsub231ps  (%eax), %xmm5, %xmm1

// CHECK: vfmadd132pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x98,0xca]
          vfmadd132pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd132pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x98,0x08]
          vfmadd132pd  (%eax), %ymm5, %ymm1

// CHECK: vfmadd132ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x98,0xca]
          vfmadd132ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd132ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x98,0x08]
          vfmadd132ps  (%eax), %ymm5, %ymm1

// CHECK: vfmadd213pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa8,0xca]
          vfmadd213pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd213pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa8,0x08]
          vfmadd213pd  (%eax), %ymm5, %ymm1

// CHECK: vfmadd213ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa8,0xca]
          vfmadd213ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd213ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa8,0x08]
          vfmadd213ps  (%eax), %ymm5, %ymm1

// CHECK: vfmadd231pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb8,0xca]
          vfmadd231pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd231pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb8,0x08]
          vfmadd231pd  (%eax), %ymm5, %ymm1

// CHECK: vfmadd231ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb8,0xca]
          vfmadd231ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmadd231ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb8,0x08]
          vfmadd231ps  (%eax), %ymm5, %ymm1

// CHECK: vfmaddsub132pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x96,0xca]
          vfmaddsub132pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmaddsub132pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x96,0x08]
          vfmaddsub132pd  (%eax), %ymm5, %ymm1

// CHECK: vfmaddsub132ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x96,0xca]
          vfmaddsub132ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmaddsub132ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x96,0x08]
          vfmaddsub132ps  (%eax), %ymm5, %ymm1

// CHECK: vfmaddsub213pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa6,0xca]
          vfmaddsub213pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmaddsub213pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa6,0x08]
          vfmaddsub213pd  (%eax), %ymm5, %ymm1

// CHECK: vfmaddsub213ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa6,0xca]
          vfmaddsub213ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmaddsub213ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa6,0x08]
          vfmaddsub213ps  (%eax), %ymm5, %ymm1

// CHECK: vfmaddsub231pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb6,0xca]
          vfmaddsub231pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmaddsub231pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb6,0x08]
          vfmaddsub231pd  (%eax), %ymm5, %ymm1

// CHECK: vfmaddsub231ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb6,0xca]
          vfmaddsub231ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmaddsub231ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb6,0x08]
          vfmaddsub231ps  (%eax), %ymm5, %ymm1

// CHECK: vfmsubadd132pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x97,0xca]
          vfmsubadd132pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmsubadd132pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x97,0x08]
          vfmsubadd132pd  (%eax), %ymm5, %ymm1

// CHECK: vfmsubadd132ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x97,0xca]
          vfmsubadd132ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmsubadd132ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x97,0x08]
          vfmsubadd132ps  (%eax), %ymm5, %ymm1

// CHECK: vfmsubadd213pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa7,0xca]
          vfmsubadd213pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmsubadd213pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xa7,0x08]
          vfmsubadd213pd  (%eax), %ymm5, %ymm1

// CHECK: vfmsubadd213ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa7,0xca]
          vfmsubadd213ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmsubadd213ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xa7,0x08]
          vfmsubadd213ps  (%eax), %ymm5, %ymm1

// CHECK: vfmsubadd231pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb7,0xca]
          vfmsubadd231pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmsubadd231pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xb7,0x08]
          vfmsubadd231pd  (%eax), %ymm5, %ymm1

// CHECK: vfmsubadd231ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb7,0xca]
          vfmsubadd231ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmsubadd231ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xb7,0x08]
          vfmsubadd231ps  (%eax), %ymm5, %ymm1

// CHECK: vfmsub132pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x9a,0xca]
          vfmsub132pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmsub132pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x9a,0x08]
          vfmsub132pd  (%eax), %ymm5, %ymm1

// CHECK: vfmsub132ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x9a,0xca]
          vfmsub132ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmsub132ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x9a,0x08]
          vfmsub132ps  (%eax), %ymm5, %ymm1

// CHECK: vfmsub213pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xaa,0xca]
          vfmsub213pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmsub213pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xaa,0x08]
          vfmsub213pd  (%eax), %ymm5, %ymm1

// CHECK: vfmsub213ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xaa,0xca]
          vfmsub213ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmsub213ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xaa,0x08]
          vfmsub213ps  (%eax), %ymm5, %ymm1

// CHECK: vfmsub231pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xba,0xca]
          vfmsub231pd  %ymm2, %ymm5, %ymm1

// CHECK: vfmsub231pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xba,0x08]
          vfmsub231pd  (%eax), %ymm5, %ymm1

// CHECK: vfmsub231ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xba,0xca]
          vfmsub231ps  %ymm2, %ymm5, %ymm1

// CHECK: vfmsub231ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xba,0x08]
          vfmsub231ps  (%eax), %ymm5, %ymm1

// CHECK: vfnmadd132pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x9c,0xca]
          vfnmadd132pd  %ymm2, %ymm5, %ymm1

// CHECK: vfnmadd132pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x9c,0x08]
          vfnmadd132pd  (%eax), %ymm5, %ymm1

// CHECK: vfnmadd132ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x9c,0xca]
          vfnmadd132ps  %ymm2, %ymm5, %ymm1

// CHECK: vfnmadd132ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x9c,0x08]
          vfnmadd132ps  (%eax), %ymm5, %ymm1

// CHECK: vfnmadd213pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xac,0xca]
          vfnmadd213pd  %ymm2, %ymm5, %ymm1

// CHECK: vfnmadd213pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xac,0x08]
          vfnmadd213pd  (%eax), %ymm5, %ymm1

// CHECK: vfnmadd213ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xac,0xca]
          vfnmadd213ps  %ymm2, %ymm5, %ymm1

// CHECK: vfnmadd213ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xac,0x08]
          vfnmadd213ps  (%eax), %ymm5, %ymm1

// CHECK: vfnmadd231pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xbc,0xca]
          vfnmadd231pd  %ymm2, %ymm5, %ymm1

// CHECK: vfnmadd231pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xbc,0x08]
          vfnmadd231pd  (%eax), %ymm5, %ymm1

// CHECK: vfnmadd231ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xbc,0xca]
          vfnmadd231ps  %ymm2, %ymm5, %ymm1

// CHECK: vfnmadd231ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xbc,0x08]
          vfnmadd231ps  (%eax), %ymm5, %ymm1

// CHECK: vfnmsub132pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x9e,0xca]
          vfnmsub132pd  %ymm2, %ymm5, %ymm1

// CHECK: vfnmsub132pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0x9e,0x08]
          vfnmsub132pd  (%eax), %ymm5, %ymm1

// CHECK: vfnmsub132ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x9e,0xca]
          vfnmsub132ps  %ymm2, %ymm5, %ymm1

// CHECK: vfnmsub132ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x9e,0x08]
          vfnmsub132ps  (%eax), %ymm5, %ymm1

// CHECK: vfnmsub213pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xae,0xca]
          vfnmsub213pd  %ymm2, %ymm5, %ymm1

// CHECK: vfnmsub213pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xae,0x08]
          vfnmsub213pd  (%eax), %ymm5, %ymm1

// CHECK: vfnmsub213ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xae,0xca]
          vfnmsub213ps  %ymm2, %ymm5, %ymm1

// CHECK: vfnmsub213ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xae,0x08]
          vfnmsub213ps  (%eax), %ymm5, %ymm1

// CHECK: vfnmsub231pd  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xbe,0xca]
          vfnmsub231pd  %ymm2, %ymm5, %ymm1

// CHECK: vfnmsub231pd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0xd5,0xbe,0x08]
          vfnmsub231pd  (%eax), %ymm5, %ymm1

// CHECK: vfnmsub231ps  %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xbe,0xca]
          vfnmsub231ps  %ymm2, %ymm5, %ymm1

// CHECK: vfnmsub231ps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0xbe,0x08]
          vfnmsub231ps  (%eax), %ymm5, %ymm1

