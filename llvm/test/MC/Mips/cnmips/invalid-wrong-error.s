# These were correctly rejected as not being supported but the wrong
# error message was emitted.
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=octeon 2>%t1
# RUN: FileCheck %s < %t1

  .set  noat
  lwc3  $4, 0($5)  # CHECK: :[[#]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
  swc3  $4, 0($5)  # CHECK: :[[#]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
  ldc3  $4, 0($5)  # CHECK: :[[#]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
  sdc3  $4, 0($5)  # CHECK: :[[#]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
