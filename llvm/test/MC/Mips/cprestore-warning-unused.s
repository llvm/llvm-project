# RUN: llvm-mc %s -triple mips-unknown-linux-gnu --position-independent 2>%t1
# RUN: FileCheck %s < %t1

  .text
  .set noreorder
  .cpload $25
  .set reorder

  jal $25
# CHECK: :[[@LINE-1]]:3: warning: no .cprestore used in PIC mode
