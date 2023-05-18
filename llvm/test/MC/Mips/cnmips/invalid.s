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
    ins $2, $3, -1, 1    # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
    ins $2, $3, 32, 1    # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
    ins $2, $3, 0, -1    # CHECK: :[[@LINE]]:20: error: expected immediate in range 1 .. 32
    ins $2, $3, 0, 33    # CHECK: :[[@LINE]]:20: error: expected immediate in range 1 .. 32
    seqi $2, $3, -1025   # CHECK: :[[@LINE]]:18: error: expected 10-bit signed immediate
    seqi $2, $3, 1024    # CHECK: :[[@LINE]]:18: error: expected 10-bit signed immediate
    snei $2, $3, -1025   # CHECK: :[[@LINE]]:18: error: expected 10-bit signed immediate
    snei $2, $3, 1024    # CHECK: :[[@LINE]]:18: error: expected 10-bit signed immediate
