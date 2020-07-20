# RUN: llvm-mc < %s -arch=mips -mcpu=mips32r2 -filetype=obj -o - | \
# RUN:   llvm-readobj -A - | FileCheck --check-prefix=CHECK-OBJ %s
# RUN: llvm-mc < %s -arch=mips -mcpu=mips32r2 -filetype=asm -o - | \
# RUN:   FileCheck --check-prefix=CHECK-ASM %s

# Test that the .module directive sets the MT flag in .MIPS.abiflags when
# assembling to boject files.

# Test that the .moodule directive is re-emitted when expanding assembly.

# CHECK-OBJ: ASEs
# CHECK-OBJ-NEXT: MT (0x40)

# CHECK-ASM:  .module mt
.module mt
nop
