# RUN: llvm-mc %s -arch=mips -mcpu=mips32r6 -mattr=+ginv | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32r6 -filetype=obj -o - -mattr=+ginv | \
# RUN:   llvm-readobj --mips-abi-flags - | \
# RUN:   FileCheck %s -check-prefix=CHECK-OBJ

# CHECK-ASM: .module noginv

# Check that MIPS.abiflags has no GINV flag.
# CHECK-OBJ: MIPS ABI Flags {
# CHECK-OBJ:   ASEs [ (0x0)
# CHECK-OBJ-NOT:   ASEs [ (0x20000)
# CHECK-OBJ-NOT:     GINV (0x20000)
# CHECK-OBJ: }

  .module noginv

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
