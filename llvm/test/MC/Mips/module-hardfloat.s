# RUN: llvm-mc %s -triple mips-unknown-linux-gnu | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -filetype=obj -o - | \
# RUN:   llvm-readobj --mips-abi-flags - | \
# RUN:     FileCheck %s -check-prefix=CHECK-OBJ

# CHECK-ASM: .module hardfloat

# Check if the MIPS.abiflags section was correctly emitted:
# CHECK-OBJ: MIPS ABI Flags {
# CHECK-OBJ:   FP ABI: Hard float (32-bit CPU, Any FPU) (0x5)
# CHECK-OBJ:   CPR1 size: 32
# CHECK-OBJ:   Flags 1 [ (0x1)
# CHECK-OBJ:     ODDSPREG (0x1)
# CHECK-OBJ:   ]
# CHECK-OBJ: }

  .module fp=xx
  .module oddspreg
  .module softfloat
  .module hardfloat

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
