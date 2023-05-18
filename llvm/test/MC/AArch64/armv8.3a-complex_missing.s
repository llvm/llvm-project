// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a,-complxnum,+fullfp16 -o - %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+fullfp16 -o - %s 2>&1 | FileCheck %s
fcmla v0.4h, v1.4h, v2.4h, #0
fcmla v0.8h, v1.8h, v2.8h, #0
fcmla v0.2s, v1.2s, v2.2s, #0
fcmla v0.4s, v1.4s, v2.4s, #0
fcmla v0.2d, v1.2d, v2.2d, #0
fcmla v0.2s, v1.2s, v2.2s, #0
fcmla v0.2s, v1.2s, v2.2s, #90
fcmla v0.2s, v1.2s, v2.2s, #180
fcmla v0.2s, v1.2s, v2.2s, #270
fcadd v0.4h, v1.4h, v2.4h, #90
fcadd v0.8h, v1.8h, v2.8h, #90
fcadd v0.2s, v1.2s, v2.2s, #90
fcadd v0.4s, v1.4s, v2.4s, #90
fcadd v0.2d, v1.2d, v2.2d, #90
fcadd v0.2s, v1.2s, v2.2s, #90
fcadd v0.2s, v1.2s, v2.2s, #270
fcmla v0.4h, v1.4h, v2.h[0], #0
fcmla v0.8h, v1.8h, v2.h[0], #0
fcmla v0.4s, v1.4s, v2.s[0], #0
fcmla v0.4s, v1.4s, v2.s[0], #90
fcmla v0.4s, v1.4s, v2.s[0], #180
fcmla v0.4s, v1.4s, v2.s[0], #270
fcmla v0.4h, v1.4h, v2.h[1], #0
fcmla v0.8h, v1.8h, v2.h[3], #0
fcmla v0.4s, v1.4s, v2.s[1], #0
//CHECK: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4h, v1.4h, v2.4h, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.8h, v1.8h, v2.8h, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.4s, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.2d, v1.2d, v2.2d, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #180
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.2s, v1.2s, v2.2s, #270
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcadd v0.4h, v1.4h, v2.4h, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcadd v0.8h, v1.8h, v2.8h, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcadd v0.4s, v1.4s, v2.4s, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcadd v0.2d, v1.2d, v2.2d, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcadd v0.2s, v1.2s, v2.2s, #270
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4h, v1.4h, v2.h[0], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.8h, v1.8h, v2.h[0], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[0], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[0], #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[0], #180
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[0], #270
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4h, v1.4h, v2.h[1], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.8h, v1.8h, v2.h[3], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}} error: instruction requires: complxnum
//CHECK-NEXT: fcmla v0.4s, v1.4s, v2.s[1], #0
//CHECK-NEXT: ^

