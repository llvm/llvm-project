# RUN: llvm-mc -preserve-comments -triple amdgcn-amd- %s >%t-1.s
# RUN: llvm-mc -preserve-comments -triple amdgcn-amd- %t-1.s >%t-2.s
# RUN: diff %t-1.s %t-2.s

# Test that AMDGPU assembly round-trips when run through MC; the first
# transition from hand-written to "canonical" output may introduce some small
# differences, so we don't include the initial input in the comparison.

.text

# The AMDGPU asm parser didn't consume the end of statement
# consistently, which led to extra empty lines in the output.
s_nop 0
