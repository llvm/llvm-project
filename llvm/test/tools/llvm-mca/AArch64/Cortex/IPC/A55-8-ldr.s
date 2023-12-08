# RUN: llvm-mca -mtriple=aarch64 -mcpu=cortex-a55 --all-views=false --summary-view --iterations=1000 < %s | FileCheck %s
# CHECK: IPC:
# CHECK-SAME: 1.50
#
# XFAIL: *
#
# MCA reports IPC = 0.60, while hardware shows IPC = 1.50.
#
# 1) The skewed ALU on Cortex-A55 is not modeled: ADD and AND
#    instructions should be issued in the same cycle.
#    See A55-2.s test for more details.
#
# 2) Cortex-A55 manual mentions that there is a forwarding path from
#    the ALU pipeline to the LD/ST pipeline. This is not implemented in
#    the LLVM scheduling model.

add	w8, w8, #1
and	w12, w8, #0x3f
ldr	w14, [x10, w12, uxtw #2]
