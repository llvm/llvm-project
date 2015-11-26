# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r2 \
# RUN:     -mattr==eva 2>%t1
# RUN: FileCheck %s < %t1

    .set noat
    cachee -1, 255($7) # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
    cachee 32, 255($7) # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
    prefe -1, 255($7)  # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
    prefe 32, 255($7)  # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
