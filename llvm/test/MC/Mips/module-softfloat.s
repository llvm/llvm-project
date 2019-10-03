# RUN: llvm-mc %s -arch=mips -mcpu=mips32 | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -filetype=obj -o - | \
# RUN:   llvm-readobj -A - | \
# RUN:     FileCheck %s -check-prefix=CHECK-OBJ

# CHECK-ASM: .module softfloat

# Check if the MIPS.abiflags section was correctly emitted:
# CHECK-OBJ: MIPS ABI Flags {
# CHECK-OBJ:   FP ABI: Soft float (0x3)
# CHECK-OBJ:   CPR1 size: 0
# CHECK-OBJ: }

  .module softfloat

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
