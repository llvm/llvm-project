# RUN: llvm-mc %s -arch=mips -mcpu=mips32r6 -mattr=+crc | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32r6 -filetype=obj -o - -mattr=+crc | \
# RUN:   llvm-readobj --mips-abi-flags - | \
# RUN:   FileCheck %s -check-prefix=CHECK-OBJ

# CHECK-ASM: .module nocrc

# Check that MIPS.abiflags has no CRC flag.
# CHECK-OBJ: MIPS ABI Flags {
# CHECK-OBJ:   ASEs [ (0x0)
# CHECK-OBJ-NOT:   ASEs [ (0x8000)
# CHECK-OBJ-NOT:     CRC (0x8000)
# CHECK-OBJ: }

  .module nocrc

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
