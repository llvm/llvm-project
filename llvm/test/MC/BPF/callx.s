# RUN: llvm-mc -triple bpfel -show-encoding < %s | FileCheck %s

# CHECK: callx r1                                # encoding: [0x8d,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
callx r1
