# RUN: not llvm-mc %s -arch=mips -mcpu=mips32 2>%t1
# RUN: FileCheck %s < %t1

  .text
  .set noreorder
  .cpload $25

  .set mips16
  .cprestore 8
# CHECK: :[[@LINE-1]]:14: error: .cprestore is not supported in Mips16 mode
  .set nomips16

  .cprestore
# CHECK: :[[@LINE-1]]:13: error: expected stack offset value

  .cprestore foo
# CHECK: :[[@LINE-1]]:17: error: stack offset is not an absolute expression

  .cprestore -8
# CHECK: :[[@LINE-1]]:3: warning: .cprestore with negative stack offset has no effect

  .cprestore 8, 35, bar
# CHECK: :[[@LINE-1]]:15: error: unexpected token, expected end of statement
