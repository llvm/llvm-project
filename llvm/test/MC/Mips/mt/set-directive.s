# RUN: llvm-mc < %s -arch=mips -mcpu=mips32r2 -filetype=obj -o - | \
# RUN:   llvm-readobj -A | FileCheck %s --check-prefix=CHECK-OBJ
# RUN: llvm-mc < %s -arch=mips -mcpu=mips32r2 -filetype=asm -o - | \
# RUN:   FileCheck %s --check-prefix=CHECK-ASM

# Test that the MT ASE flag in .MIPS.abiflags is _not_ set by .set.
# Test that '.set mt' is emitted by the asm target streamer.

# CHECK-OBJ: ASEs
# CHECK-OBJ-NOT: MT (0x40)

# CHECK-ASM: .set mt
  .set  mt
  nop
