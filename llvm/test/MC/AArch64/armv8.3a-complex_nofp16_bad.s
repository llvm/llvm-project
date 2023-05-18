// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+complxnum -o - %s 2>&1 | FileCheck %s
fcmla v0.4h, v1.4h, v2.4h, #0
fcmla v0.8h, v1.8h, v2.8h, #0
fcadd v0.4h, v1.4h, v2.4h, #90
fcadd v0.8h, v1.8h, v2.8h, #90
fcmla v0.4h, v1.4h, v2.h[0], #0
fcmla v0.8h, v1.8h, v2.h[0], #0
fcmla v0.4h, v1.4h, v2.h[1], #0
fcmla v0.8h, v1.8h, v2.h[3], #0
//CHECK: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcmla v0.4h, v1.4h, v2.4h, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcmla v0.8h, v1.8h, v2.8h, #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcadd v0.4h, v1.4h, v2.4h, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcadd v0.8h, v1.8h, v2.8h, #90
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcmla v0.4h, v1.4h, v2.h[0], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcmla v0.8h, v1.8h, v2.h[0], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcmla v0.4h, v1.4h, v2.h[1], #0
//CHECK-NEXT: ^
//CHECK-NEXT: {{.*}}error: instruction requires: fullfp16
//CHECK-NEXT: fcmla v0.8h, v1.8h, v2.h[3], #0
//CHECK-NEXT: ^

