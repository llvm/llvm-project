# RUN: not llvm-mc %s -triple=mips -show-encoding -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  break16 -1 # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  break16 16 # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  sdbbp16 -1 # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  sdbbp16 16 # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
