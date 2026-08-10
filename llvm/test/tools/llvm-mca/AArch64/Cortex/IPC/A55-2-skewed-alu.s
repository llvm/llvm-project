# RUN: llvm-mca -mtriple=aarch64 -mcpu=cortex-a55 --all-views=false --summary-view --iterations=1000 < %s | FileCheck %s
# CHECK: IPC:
# CHECK-SAME: 2.00
#
# XFAIL: *
#
# Cortex-A55 has a secondary skewed ALU in the Ex1 stage for simple
# ALU instructions that do not require shifting or saturation
# resources. Results from the skewed ALU are available 1 cycle earlier.
#
# This features allows the first and the second instruction to be
# dual-issued despite a register dependency (w8).
#
# MCA and LLVM scheduling model do not support this yet.

add	w8, w8, #1
add	w10, w8, #1
add	w12, w8, #1
