# RUN: not llvm-mc %s -mcpu=mips32 -mattr=+dspr2 -triple mips-unknown-linux 2>%t1
# RUN: FileCheck %s < %t1

  lbux    $7, $10($11)
  append  $4, $10, 2

  .set nodsp
  lbux    $6, $10($11)
  # CHECK: error: instruction requires a CPU feature not currently enabled

  .set dsp
  lbux    $5, $10($11)
  # CHECK-NOT: error: instruction requires a CPU feature not currently enabled

  .set nodsp
  append  $3, $10, 2
  # CHECK: error: instruction requires a CPU feature not currently enabled

  .set dspr2
  append  $2, $10, 2
  # CHECK-NOT: error: instruction requires a CPU feature not currently enabled
