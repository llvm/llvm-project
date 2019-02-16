// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+complxnum,+fullfp16 -o - %s 2>&1 | FileCheck %s
fcmla v0.2s, v1.2s, v2.2s, #1
fcmla v0.2s, v1.2s, v2.2s, #360
fcmla v0.2s, v1.2s, v2.2s, #-90
fcadd v0.2s, v1.2s, v2.2s, #1
fcadd v0.2s, v1.2s, v2.2s, #360
fcadd v0.2s, v1.2s, v2.2s, #-90
fcadd v0.2s, v1.2s, v2.2s, #0
fcadd v0.2s, v1.2s, v2.2s, #180
fcmla v0.4h, v1.4h, v2.h[2], #0
fcmla v0.8h, v1.8h, v2.h[4], #0
fcmla v0.4s, v1.4s, v2.s[2], #0
fcmla v0.4s, v1.4s, v2.s[0], #1
fcmla v0.4s, v1.4s, v2.s[0], #360
fcmla v0.4s, v1.4s, v2.s[0], #-90
//CHECK: {{.*}}error: complex rotation must be 0, 90, 180 or 270.
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #1
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 0, 90, 180 or 270.
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #360
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 0, 90, 180 or 270.
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #-90
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 90 or 270.
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #1
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 90 or 270.
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #360
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 90 or 270.
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #-90
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 90 or 270.
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #0
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 90 or 270.
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #180
//CHECK-NEXT:                            ^
//CHECK-NEXT: {{.*}}error: vector lane must be an integer in range [0, 1].
//CHECK-NEXT: fcmla v0.4h, v1.4h, v2.h[2], #0
//CHECK-NEXT:                         ^
//CHECK-NEXT: {{.*}}error: vector lane must be an integer in range [0, 3].
//CHECK-NEXT: fcmla v0.8h, v1.8h, v2.h[4], #0
//CHECK-NEXT:                         ^
//CHECK-NEXT: {{.*}}error: vector lane must be an integer in range [0, 1].
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[2], #0
//CHECK-NEXT:                         ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 0, 90, 180 or 270.
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[0], #1
//CHECK-NEXT:                              ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 0, 90, 180 or 270.
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[0], #360
//CHECK-NEXT:                              ^
//CHECK-NEXT: {{.*}}error: complex rotation must be 0, 90, 180 or 270.
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[0], #-90
//CHECK-NEXT:                              ^

