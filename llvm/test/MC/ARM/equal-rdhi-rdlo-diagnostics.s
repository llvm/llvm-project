@ RUN: not llvm-mc -triple thumbv7m-eabi -mattr=+dsp < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple armv8 -mattr=+dsp < %s 2>&1 | FileCheck %s

  smlal   r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlalbb   r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlalbt   r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlaltb   r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlaltt   r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlald   r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlaldx   r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlsld  r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smlsldx  r1, r1, r3, r4
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  smull   r1, r1, r2, r3
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  umull   r1, r1, r2, r3
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  umlal   r1, r1, r2, r3
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
  umaal   r1, r1, r2, r3
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable instruction, RdHi and RdLo must be different
