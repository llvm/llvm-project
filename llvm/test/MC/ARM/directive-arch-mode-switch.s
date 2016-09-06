@ RUN: llvm-mc -triple arm-none-eabi -filetype asm %s 2>%t | FileCheck %s
@ RUN: FileCheck %s <%t --check-prefix=STDERR

@ Start in arm mode
  .arm
@ CHECK: .code 32

@ In ARM mode, switch to an arch which has ARM and Thumb, no warning or .code directive (stay in ARM mode)
  .arch armv7-a
@ STDERR-NOT: [[@LINE-1]]:{{[0-9]+}}: warning:
@ CHECK-NOT: .code
@ CHECK: .arch   armv7-a
@ CHECK-NOT: .code

@ In ARM mode, switch to an arch which has Thumb only, expect warning and .code 16 directive
  .arch armv6-m
@ STDERR: [[@LINE-1]]:{{[0-9]+}}: warning: new target does not support arm mode, switching to thumb mode
@ CHECK: .code   16
@ CHECK: .arch   armv6-m

@ In Thumb mode, switch to an arch which has ARM and Thumb, no warning or .code directive (stay in Thumb mode)
  .arch armv7-a
@ STDERR-NOT: [[@LINE-1]]:{{[0-9]+}}: warning:
@ CHECK-NOT: .code
@ CHECK: .arch   armv7-a
@ CHECK-NOT: .code

@ In Thumb mode, switch to a CPU which has ARM and Thumb, no warning or .code directive (stay in Thumb mode)
  .cpu cortex-a8
@ STDERR-NOT: [[@LINE-1]]:{{[0-9]+}}: warning:
@ CHECK-NOT: .code
@ CHECK: .cpu cortex-a8
@ CHECK-NOT: .code

@ Switch to ARM mode
  .arm
@ CHECK: .code 32

@ In ARM mode, switch to a CPU which has ARM and Thumb, no warning or .code directive (stay in ARM mode)
  .cpu cortex-a8
@ STDERR-NOT: [[@LINE-1]]:{{[0-9]+}}: warning:
@ CHECK-NOT: .code
@ CHECK: .cpu cortex-a8
@ CHECK-NOT: .code

@ In ARM mode, switch to a CPU which has Thumb only, expect warning and .code 16 directive
  .cpu cortex-m3
@ STDERR: [[@LINE-1]]:{{[0-9]+}}: warning: new target does not support arm mode, switching to thumb mode
@ CHECK: .cpu    cortex-m3
@ CHECK: .code   16

@ We don't have any ARM-only targets (i.e. v4), so we can't test the forced Thumb->ARM case
