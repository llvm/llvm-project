# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=micromips 2>&1 | FileCheck %s

  .text
  .type main, @function
  .set micromips
main:
# CHECK-NOT: warning: macro instruction expanded into multiple instructions
  .set macro
  b 132
  b 1332
  b bar

  lwm $16, $17, 8($sp)
  swm $16, $17, 8($sp)

  add $4, $5, $6

  .set noreorder
  .set nomacro
  b 132
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  b 1332
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  b bar
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions

  lwm $16, $17, 8($sp)
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  swm $16, $17, 8($sp)
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions

  add $4, $5, $6
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
