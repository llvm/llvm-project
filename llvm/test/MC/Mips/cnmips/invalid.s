# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=octeon 2>%t1
# RUN: FileCheck %s < %t1

    .set noat
foo:
    bbit0 $19, -1, foo   # CHECK: :[[@LINE]]:16: error: expected 6-bit unsigned immediate
    bbit0 $19, 64, foo   # CHECK: :[[@LINE]]:16: error: expected 6-bit unsigned immediate
    bbit032 $19, -1, foo # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
    bbit032 $19, 32, foo # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
    bbit1 $19, -1, foo   # CHECK: :[[@LINE]]:16: error: expected 6-bit unsigned immediate
    bbit1 $19, 64, foo   # CHECK: :[[@LINE]]:16: error: expected 6-bit unsigned immediate
    bbit132 $19, -1, foo # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
    bbit132 $19, 32, foo # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
