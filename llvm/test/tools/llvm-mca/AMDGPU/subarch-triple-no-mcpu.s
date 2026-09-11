# With an AMDGPU subarch triple and no -mcpu, llvm-mca must resolve
# the processor from the triple's subarch field rather than analyzing
# with no CPU.
#
# RUN: llvm-mca -mtriple=amdgpu9.0a --iterations=1 < %s | FileCheck %s

v_add_f64 v[0:1], v[0:1], v[2:3]

# CHECK: Iterations:        1
# CHECK: Instructions:      1
