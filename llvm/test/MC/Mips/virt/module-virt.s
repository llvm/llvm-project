# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips32r5 | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips32r5 \
# RUN:   -filetype=obj -o - | \
# RUN:   llvm-readobj -A - | \
# RUN:   FileCheck %s -check-prefix=CHECK-OBJ

# CHECK-ASM: .module virt

# Check if the MIPS.abiflags section was correctly emitted:
# CHECK-OBJ: MIPS ABI Flags {
# CHECK-OBJ:   ASEs [ (0x100)
# CHECK-OBJ:     VZ (0x100)
# CHECK-OBJ: }

  .module virt
  hypcall

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
