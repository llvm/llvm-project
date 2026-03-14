// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vfmadd132pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0x98,0xdc]
          vfmadd132pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd132pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0x98,0x18]
          vfmadd132pd  (%rax), %xmm10, %xmm11

// CHECK: vfmadd132ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x98,0xdc]
          vfmadd132ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd132ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0x98,0x18]
          vfmadd132ps  (%rax), %xmm10, %xmm11

// CHECK: vfmadd213pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xa8,0xdc]
          vfmadd213pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd213pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xa8,0x18]
          vfmadd213pd  (%rax), %xmm10, %xmm11

// CHECK: vfmadd213ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xa8,0xdc]
          vfmadd213ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd213ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xa8,0x18]
          vfmadd213ps  (%rax), %xmm10, %xmm11

// CHECK: vfmadd231pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xb8,0xdc]
          vfmadd231pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd231pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xb8,0x18]
          vfmadd231pd  (%rax), %xmm10, %xmm11

// CHECK: vfmadd231ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xb8,0xdc]
          vfmadd231ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd231ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xb8,0x18]
          vfmadd231ps  (%rax), %xmm10, %xmm11

// CHECK: vfmadd132pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0x98,0xdc]
          vfmadd132pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd132pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0x98,0x18]
          vfmadd132pd  (%rax), %ymm10, %ymm11

// CHECK: vfmadd132ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x98,0xdc]
          vfmadd132ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd132ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x98,0x18]
          vfmadd132ps  (%rax), %ymm10, %ymm11

// CHECK: vfmadd213pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xa8,0xdc]
          vfmadd213pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd213pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xa8,0x18]
          vfmadd213pd  (%rax), %ymm10, %ymm11

// CHECK: vfmadd213ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xa8,0xdc]
          vfmadd213ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd213ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xa8,0x18]
          vfmadd213ps  (%rax), %ymm10, %ymm11

// CHECK: vfmadd231pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xb8,0xdc]
          vfmadd231pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd231pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xb8,0x18]
          vfmadd231pd  (%rax), %ymm10, %ymm11

// CHECK: vfmadd231ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xb8,0xdc]
          vfmadd231ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd231ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xb8,0x18]
          vfmadd231ps  (%rax), %ymm10, %ymm11

// CHECK: vfmadd132pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0x98,0xdc]
          vfmadd132pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd132pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0x98,0x18]
          vfmadd132pd  (%rax), %xmm10, %xmm11

// CHECK: vfmadd132ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x98,0xdc]
          vfmadd132ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd132ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0x98,0x18]
          vfmadd132ps  (%rax), %xmm10, %xmm11

// CHECK: vfmadd213pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xa8,0xdc]
          vfmadd213pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd213pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xa8,0x18]
          vfmadd213pd  (%rax), %xmm10, %xmm11

// CHECK: vfmadd213ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xa8,0xdc]
          vfmadd213ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd213ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xa8,0x18]
          vfmadd213ps  (%rax), %xmm10, %xmm11

// CHECK: vfmadd231pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xb8,0xdc]
          vfmadd231pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd231pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xb8,0x18]
          vfmadd231pd  (%rax), %xmm10, %xmm11

// CHECK: vfmadd231ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xb8,0xdc]
          vfmadd231ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmadd231ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xb8,0x18]
          vfmadd231ps  (%rax), %xmm10, %xmm11

// CHECK: vfmaddsub132pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0x96,0xdc]
          vfmaddsub132pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmaddsub132pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0x96,0x18]
          vfmaddsub132pd  (%rax), %xmm10, %xmm11

// CHECK: vfmaddsub132ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x96,0xdc]
          vfmaddsub132ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmaddsub132ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0x96,0x18]
          vfmaddsub132ps  (%rax), %xmm10, %xmm11

// CHECK: vfmaddsub213pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xa6,0xdc]
          vfmaddsub213pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmaddsub213pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xa6,0x18]
          vfmaddsub213pd  (%rax), %xmm10, %xmm11

// CHECK: vfmaddsub213ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xa6,0xdc]
          vfmaddsub213ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmaddsub213ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xa6,0x18]
          vfmaddsub213ps  (%rax), %xmm10, %xmm11

// CHECK: vfmaddsub231pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xb6,0xdc]
          vfmaddsub231pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmaddsub231pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xb6,0x18]
          vfmaddsub231pd  (%rax), %xmm10, %xmm11

// CHECK: vfmaddsub231ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xb6,0xdc]
          vfmaddsub231ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmaddsub231ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xb6,0x18]
          vfmaddsub231ps  (%rax), %xmm10, %xmm11

// CHECK: vfmsubadd132pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0x97,0xdc]
          vfmsubadd132pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmsubadd132pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0x97,0x18]
          vfmsubadd132pd  (%rax), %xmm10, %xmm11

// CHECK: vfmsubadd132ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x97,0xdc]
          vfmsubadd132ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmsubadd132ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0x97,0x18]
          vfmsubadd132ps  (%rax), %xmm10, %xmm11

// CHECK: vfmsubadd213pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xa7,0xdc]
          vfmsubadd213pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmsubadd213pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xa7,0x18]
          vfmsubadd213pd  (%rax), %xmm10, %xmm11

// CHECK: vfmsubadd213ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xa7,0xdc]
          vfmsubadd213ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmsubadd213ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xa7,0x18]
          vfmsubadd213ps  (%rax), %xmm10, %xmm11

// CHECK: vfmsubadd231pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xb7,0xdc]
          vfmsubadd231pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmsubadd231pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xb7,0x18]
          vfmsubadd231pd  (%rax), %xmm10, %xmm11

// CHECK: vfmsubadd231ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xb7,0xdc]
          vfmsubadd231ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmsubadd231ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xb7,0x18]
          vfmsubadd231ps  (%rax), %xmm10, %xmm11

// CHECK: vfmsub132pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0x9a,0xdc]
          vfmsub132pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmsub132pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0x9a,0x18]
          vfmsub132pd  (%rax), %xmm10, %xmm11

// CHECK: vfmsub132ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x9a,0xdc]
          vfmsub132ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmsub132ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0x9a,0x18]
          vfmsub132ps  (%rax), %xmm10, %xmm11

// CHECK: vfmsub213pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xaa,0xdc]
          vfmsub213pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmsub213pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xaa,0x18]
          vfmsub213pd  (%rax), %xmm10, %xmm11

// CHECK: vfmsub213ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xaa,0xdc]
          vfmsub213ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmsub213ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xaa,0x18]
          vfmsub213ps  (%rax), %xmm10, %xmm11

// CHECK: vfmsub231pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xba,0xdc]
          vfmsub231pd  %xmm12, %xmm10, %xmm11

// CHECK: vfmsub231pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xba,0x18]
          vfmsub231pd  (%rax), %xmm10, %xmm11

// CHECK: vfmsub231ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xba,0xdc]
          vfmsub231ps  %xmm12, %xmm10, %xmm11

// CHECK: vfmsub231ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xba,0x18]
          vfmsub231ps  (%rax), %xmm10, %xmm11

// CHECK: vfnmadd132pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0x9c,0xdc]
          vfnmadd132pd  %xmm12, %xmm10, %xmm11

// CHECK: vfnmadd132pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0x9c,0x18]
          vfnmadd132pd  (%rax), %xmm10, %xmm11

// CHECK: vfnmadd132ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x9c,0xdc]
          vfnmadd132ps  %xmm12, %xmm10, %xmm11

// CHECK: vfnmadd132ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0x9c,0x18]
          vfnmadd132ps  (%rax), %xmm10, %xmm11

// CHECK: vfnmadd213pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xac,0xdc]
          vfnmadd213pd  %xmm12, %xmm10, %xmm11

// CHECK: vfnmadd213pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xac,0x18]
          vfnmadd213pd  (%rax), %xmm10, %xmm11

// CHECK: vfnmadd213ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xac,0xdc]
          vfnmadd213ps  %xmm12, %xmm10, %xmm11

// CHECK: vfnmadd213ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xac,0x18]
          vfnmadd213ps  (%rax), %xmm10, %xmm11

// CHECK: vfnmadd231pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xbc,0xdc]
          vfnmadd231pd  %xmm12, %xmm10, %xmm11

// CHECK: vfnmadd231pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xbc,0x18]
          vfnmadd231pd  (%rax), %xmm10, %xmm11

// CHECK: vfnmadd231ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xbc,0xdc]
          vfnmadd231ps  %xmm12, %xmm10, %xmm11

// CHECK: vfnmadd231ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xbc,0x18]
          vfnmadd231ps  (%rax), %xmm10, %xmm11

// CHECK: vfnmsub132pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0x9e,0xdc]
          vfnmsub132pd  %xmm12, %xmm10, %xmm11

// CHECK: vfnmsub132pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0x9e,0x18]
          vfnmsub132pd  (%rax), %xmm10, %xmm11

// CHECK: vfnmsub132ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x9e,0xdc]
          vfnmsub132ps  %xmm12, %xmm10, %xmm11

// CHECK: vfnmsub132ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0x9e,0x18]
          vfnmsub132ps  (%rax), %xmm10, %xmm11

// CHECK: vfnmsub213pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xae,0xdc]
          vfnmsub213pd  %xmm12, %xmm10, %xmm11

// CHECK: vfnmsub213pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xae,0x18]
          vfnmsub213pd  (%rax), %xmm10, %xmm11

// CHECK: vfnmsub213ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xae,0xdc]
          vfnmsub213ps  %xmm12, %xmm10, %xmm11

// CHECK: vfnmsub213ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xae,0x18]
          vfnmsub213ps  (%rax), %xmm10, %xmm11

// CHECK: vfnmsub231pd  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0xa9,0xbe,0xdc]
          vfnmsub231pd  %xmm12, %xmm10, %xmm11

// CHECK: vfnmsub231pd  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0xa9,0xbe,0x18]
          vfnmsub231pd  (%rax), %xmm10, %xmm11

// CHECK: vfnmsub231ps  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xbe,0xdc]
          vfnmsub231ps  %xmm12, %xmm10, %xmm11

// CHECK: vfnmsub231ps  (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x62,0x29,0xbe,0x18]
          vfnmsub231ps  (%rax), %xmm10, %xmm11

// CHECK: vfmadd132pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0x98,0xdc]
          vfmadd132pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd132pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0x98,0x18]
          vfmadd132pd  (%rax), %ymm10, %ymm11

// CHECK: vfmadd132ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x98,0xdc]
          vfmadd132ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd132ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x98,0x18]
          vfmadd132ps  (%rax), %ymm10, %ymm11

// CHECK: vfmadd213pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xa8,0xdc]
          vfmadd213pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd213pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xa8,0x18]
          vfmadd213pd  (%rax), %ymm10, %ymm11

// CHECK: vfmadd213ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xa8,0xdc]
          vfmadd213ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd213ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xa8,0x18]
          vfmadd213ps  (%rax), %ymm10, %ymm11

// CHECK: vfmadd231pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xb8,0xdc]
          vfmadd231pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd231pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xb8,0x18]
          vfmadd231pd  (%rax), %ymm10, %ymm11

// CHECK: vfmadd231ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xb8,0xdc]
          vfmadd231ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmadd231ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xb8,0x18]
          vfmadd231ps  (%rax), %ymm10, %ymm11

// CHECK: vfmaddsub132pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0x96,0xdc]
          vfmaddsub132pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmaddsub132pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0x96,0x18]
          vfmaddsub132pd  (%rax), %ymm10, %ymm11

// CHECK: vfmaddsub132ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x96,0xdc]
          vfmaddsub132ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmaddsub132ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x96,0x18]
          vfmaddsub132ps  (%rax), %ymm10, %ymm11

// CHECK: vfmaddsub213pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xa6,0xdc]
          vfmaddsub213pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmaddsub213pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xa6,0x18]
          vfmaddsub213pd  (%rax), %ymm10, %ymm11

// CHECK: vfmaddsub213ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xa6,0xdc]
          vfmaddsub213ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmaddsub213ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xa6,0x18]
          vfmaddsub213ps  (%rax), %ymm10, %ymm11

// CHECK: vfmaddsub231pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xb6,0xdc]
          vfmaddsub231pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmaddsub231pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xb6,0x18]
          vfmaddsub231pd  (%rax), %ymm10, %ymm11

// CHECK: vfmaddsub231ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xb6,0xdc]
          vfmaddsub231ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmaddsub231ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xb6,0x18]
          vfmaddsub231ps  (%rax), %ymm10, %ymm11

// CHECK: vfmsubadd132pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0x97,0xdc]
          vfmsubadd132pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmsubadd132pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0x97,0x18]
          vfmsubadd132pd  (%rax), %ymm10, %ymm11

// CHECK: vfmsubadd132ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x97,0xdc]
          vfmsubadd132ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmsubadd132ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x97,0x18]
          vfmsubadd132ps  (%rax), %ymm10, %ymm11

// CHECK: vfmsubadd213pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xa7,0xdc]
          vfmsubadd213pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmsubadd213pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xa7,0x18]
          vfmsubadd213pd  (%rax), %ymm10, %ymm11

// CHECK: vfmsubadd213ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xa7,0xdc]
          vfmsubadd213ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmsubadd213ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xa7,0x18]
          vfmsubadd213ps  (%rax), %ymm10, %ymm11

// CHECK: vfmsubadd231pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xb7,0xdc]
          vfmsubadd231pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmsubadd231pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xb7,0x18]
          vfmsubadd231pd  (%rax), %ymm10, %ymm11

// CHECK: vfmsubadd231ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xb7,0xdc]
          vfmsubadd231ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmsubadd231ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xb7,0x18]
          vfmsubadd231ps  (%rax), %ymm10, %ymm11

// CHECK: vfmsub132pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0x9a,0xdc]
          vfmsub132pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmsub132pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0x9a,0x18]
          vfmsub132pd  (%rax), %ymm10, %ymm11

// CHECK: vfmsub132ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x9a,0xdc]
          vfmsub132ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmsub132ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x9a,0x18]
          vfmsub132ps  (%rax), %ymm10, %ymm11

// CHECK: vfmsub213pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xaa,0xdc]
          vfmsub213pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmsub213pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xaa,0x18]
          vfmsub213pd  (%rax), %ymm10, %ymm11

// CHECK: vfmsub213ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xaa,0xdc]
          vfmsub213ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmsub213ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xaa,0x18]
          vfmsub213ps  (%rax), %ymm10, %ymm11

// CHECK: vfmsub231pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xba,0xdc]
          vfmsub231pd  %ymm12, %ymm10, %ymm11

// CHECK: vfmsub231pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xba,0x18]
          vfmsub231pd  (%rax), %ymm10, %ymm11

// CHECK: vfmsub231ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xba,0xdc]
          vfmsub231ps  %ymm12, %ymm10, %ymm11

// CHECK: vfmsub231ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xba,0x18]
          vfmsub231ps  (%rax), %ymm10, %ymm11

// CHECK: vfnmadd132pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0x9c,0xdc]
          vfnmadd132pd  %ymm12, %ymm10, %ymm11

// CHECK: vfnmadd132pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0x9c,0x18]
          vfnmadd132pd  (%rax), %ymm10, %ymm11

// CHECK: vfnmadd132ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x9c,0xdc]
          vfnmadd132ps  %ymm12, %ymm10, %ymm11

// CHECK: vfnmadd132ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x9c,0x18]
          vfnmadd132ps  (%rax), %ymm10, %ymm11

// CHECK: vfnmadd213pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xac,0xdc]
          vfnmadd213pd  %ymm12, %ymm10, %ymm11

// CHECK: vfnmadd213pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xac,0x18]
          vfnmadd213pd  (%rax), %ymm10, %ymm11

// CHECK: vfnmadd213ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xac,0xdc]
          vfnmadd213ps  %ymm12, %ymm10, %ymm11

// CHECK: vfnmadd213ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xac,0x18]
          vfnmadd213ps  (%rax), %ymm10, %ymm11

// CHECK: vfnmadd231pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xbc,0xdc]
          vfnmadd231pd  %ymm12, %ymm10, %ymm11

// CHECK: vfnmadd231pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xbc,0x18]
          vfnmadd231pd  (%rax), %ymm10, %ymm11

// CHECK: vfnmadd231ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xbc,0xdc]
          vfnmadd231ps  %ymm12, %ymm10, %ymm11

// CHECK: vfnmadd231ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xbc,0x18]
          vfnmadd231ps  (%rax), %ymm10, %ymm11

// CHECK: vfnmsub132pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0x9e,0xdc]
          vfnmsub132pd  %ymm12, %ymm10, %ymm11

// CHECK: vfnmsub132pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0x9e,0x18]
          vfnmsub132pd  (%rax), %ymm10, %ymm11

// CHECK: vfnmsub132ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x9e,0xdc]
          vfnmsub132ps  %ymm12, %ymm10, %ymm11

// CHECK: vfnmsub132ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x9e,0x18]
          vfnmsub132ps  (%rax), %ymm10, %ymm11

// CHECK: vfnmsub213pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xae,0xdc]
          vfnmsub213pd  %ymm12, %ymm10, %ymm11

// CHECK: vfnmsub213pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xae,0x18]
          vfnmsub213pd  (%rax), %ymm10, %ymm11

// CHECK: vfnmsub213ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xae,0xdc]
          vfnmsub213ps  %ymm12, %ymm10, %ymm11

// CHECK: vfnmsub213ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xae,0x18]
          vfnmsub213ps  (%rax), %ymm10, %ymm11

// CHECK: vfnmsub231pd  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0xad,0xbe,0xdc]
          vfnmsub231pd  %ymm12, %ymm10, %ymm11

// CHECK: vfnmsub231pd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0xad,0xbe,0x18]
          vfnmsub231pd  (%rax), %ymm10, %ymm11

// CHECK: vfnmsub231ps  %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0xbe,0xdc]
          vfnmsub231ps  %ymm12, %ymm10, %ymm11

// CHECK: vfnmsub231ps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0xbe,0x18]
          vfnmsub231ps  (%rax), %ymm10, %ymm11

