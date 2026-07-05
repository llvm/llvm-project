## Test llvm-mc could handle .attribute correctly.

# RUN: llvm-mc %s -triple=csky -filetype=asm | FileCheck %s

.csky_attribute CSKY_ARCH_NAME, "ck810"
# CHECK: attribute      4, "ck810"

.csky_attribute CSKY_CPU_NAME, "ck810"
# CHECK: attribute      5, "ck810"

.csky_attribute CSKY_ISA_FLAGS, 0x333f
# CHECK: attribute      6, 13119 

.csky_attribute CSKY_ISA_EXT_FLAGS, 0x333f
# CHECK: attribute      7, 13119 

.csky_attribute CSKY_DSP_VERSION, 1
# CHECK: attribute      8, 1 

.csky_attribute CSKY_VDSP_VERSION, 1
# CHECK: attribute      9, 1 

.csky_attribute CSKY_FPU_VERSION, 1
# CHECK: attribute      16, 1 

.csky_attribute CSKY_FPU_ABI, 1
# CHECK: attribute      17, 1 

.csky_attribute CSKY_FPU_ROUNDING, 1
# CHECK: attribute      18, 1 

.csky_attribute CSKY_FPU_DENORMAL, 1
# CHECK: attribute      19, 1 

.csky_attribute CSKY_FPU_EXCEPTION, 1
# CHECK: attribute      20, 1 

.csky_attribute CSKY_FPU_NUMBER_MODULE, "IEEE 754"
# CHECK: attribute      21, "IEEE 754"

.csky_attribute CSKY_FPU_HARDFP, 1
# CHECK: attribute      22, 1 
