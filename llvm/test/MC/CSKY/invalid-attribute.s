## Negative tests:
##  - Feed integer value to string type attribute.
##  - Feed string value to integer type attribute.
##  - Invalid arch string.

# RUN: not llvm-mc %s -triple=csky -filetype=asm 2>&1 | FileCheck %s

.csky_attribute CSKY_ARCH_NAME, "foo"
# CHECK: [[@LINE-1]]:33: error: unknown arch name 

.csky_attribute CSKY_CPU_NAME, "foo"
# CHECK: [[@LINE-1]]:32: error: unknown cpu name

.csky_attribute CSKY_DSP_VERSION, "1"
# CHECK: [[@LINE-1]]:35: error: expected numeric constant

.csky_attribute CSKY_VDSP_VERSION, "1"
# CHECK: [[@LINE-1]]:36: error: expected numeric constant

.csky_attribute CSKY_FPU_VERSION, "1"
# CHECK: [[@LINE-1]]:35: error: expected numeric constant

.csky_attribute CSKY_FPU_ABI, "1"
# CHECK: [[@LINE-1]]:31: error: expected numeric constant

.csky_attribute CSKY_FPU_ROUNDING, "1"
# CHECK: [[@LINE-1]]:36: error: expected numeric constant

.csky_attribute CSKY_FPU_DENORMAL, "1"
# CHECK: [[@LINE-1]]:36: error: expected numeric constant

.csky_attribute CSKY_FPU_EXCEPTION, "1"
# CHECK: [[@LINE-1]]:37: error: expected numeric constant

.csky_attribute CSKY_FPU_NUMBER_MODULE, 4
# CHECK: [[@LINE-1]]:41: error: expected string constant

.csky_attribute CSKY_FPU_HARDFP, "1"
# CHECK: [[@LINE-1]]:34: error: expected numeric constant
