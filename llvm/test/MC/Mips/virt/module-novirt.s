# RUN: llvm-mc %s -arch=mips -mcpu=mips32r5 -mattr=+virt | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32r5 -filetype=obj -o - -mattr=+virt | \
# RUN:   llvm-readobj -A - | \
# RUN:   FileCheck %s -check-prefix=CHECK-OBJ

# CHECK-ASM: .module novirt

# Check that MIPS.abiflags has no VZ flag.
# CHECK-OBJ: MIPS ABI Flags {
# CHECK-OBJ:   ASEs [ (0x0)
# CHECK-OBJ-NOT:   ASEs [ (0x100)
# CHECK-OBJ-NOT:     VZ (0x100)
# CHECK-OBJ: }

  .module novirt

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
