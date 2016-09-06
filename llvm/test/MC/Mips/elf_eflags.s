# These *MUST* match the output of 'gcc -c' compiled with the same triple and
# corresponding options (-mcpu=mips32 -> -mips32 for example).

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r6 -target-abi n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R6 %s
# MIPSEL-MIPS64R6: Flags [ (0xA0000406)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r6 -target-abi n64 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R6-NAN2008 %s
# MIPSEL-MIPS64R6-NAN2008: Flags [ (0xA0000406)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r2 -target-abi n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r3 -target-abi n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r5 -target-abi n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R2 %s
# MIPSEL-MIPS64R2: Flags [ (0x80000006)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r2 -target-abi n64 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R2-NAN2008 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r3 -target-abi n64 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R2-NAN2008 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r5 -target-abi n64 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64R2-NAN2008 %s
# MIPSEL-MIPS64R2-NAN2008: Flags [ (0x80000406)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64 -target-abi n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64 %s
# MIPSEL-MIPS64: Flags [ (0x60000006)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64 -target-abi n64 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS64-NAN2008 %s
# MIPSEL-MIPS64-NAN2008: Flags [ (0x60000406)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r6 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R6 %s
# MIPSEL-MIPS32R6: Flags [ (0x90001404)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r6 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R6-NAN2008 %s
# MIPSEL-MIPS32R6-NAN2008: Flags [ (0x90001404)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r2 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r3 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r5 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R2 %s
# MIPSEL-MIPS32R2: Flags [ (0x70001004)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R2-NAN2008 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r3 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R2-NAN2008 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32r5 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32R2-NAN2008 %s
# MIPSEL-MIPS32R2-NAN2008: Flags [ (0x70001404)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32 %s
# MIPSEL-MIPS32: Flags [ (0x50001004)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS32-NAN2008 %s
# MIPSEL-MIPS32-NAN2008: Flags [ (0x50001404)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips64r2 -target-abi n32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-N32 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi n32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-N32 %s
# MIPS64EL-MIPS64R2-N32: Flags [ (0x80000024)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi n32 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-N32-NAN2008 %s
# MIPS64EL-MIPS64R2-N32-NAN2008: Flags [ (0x80000424)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 -target-abi n32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-N32 %s
# MIPS64EL-MIPS64-N32: Flags [ (0x60000024)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 -target-abi n32 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-N32-NAN2008 %s
# MIPS64EL-MIPS64-N32-NAN2008: Flags [ (0x60000424)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-N64 %s
# MIPS64EL-MIPS64R2-N64: Flags [ (0x80000006)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi n64 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-N64-NAN2008 %s
# MIPS64EL-MIPS64R2-N64-NAN2008: Flags [ (0x80000406)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -target-abi n64 -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-N64 %s
# MIPS64EL-MIPS64-N64: Flags [ (0x60000006)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -target-abi n64 -mattr=+nan2008 -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-N64-NAN2008 %s
# MIPS64EL-MIPS64-N64-NAN2008: Flags [ (0x60000406)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi o32 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-O32 %s
# MIPS64EL-MIPS64R2-O32: Flags [ (0x80001104)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi o32 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-O32-NAN2008 %s
# MIPS64EL-MIPS64R2-O32-NAN2008: Flags [ (0x80001504)

# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux -mcpu=mips5 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS5 %s
# MIPS5: Flags [ (0x40000006)

 # RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux -mcpu=mips5 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS5-NAN2008 %s
# MIPS5-NAN2008: Flags [ (0x40000406)

# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux -mcpu=mips4 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS4 %s
# MIPS4: Flags [ (0x30000006)

 # RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux -mcpu=mips4 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS4-NAN2008 %s
# MIPS4-NAN2008: Flags [ (0x30000406)

# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux -mcpu=mips3 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS3 %s
# MIPS3: Flags [ (0x20000006)

 # RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux -mcpu=mips3 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS3-NAN2008 %s
# MIPS3-NAN2008: Flags [ (0x20000406)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips2 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS2 %s
# MIPSEL-MIPS2: Flags [ (0x10001004)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips2 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-MIPS2-NAN2008 %s
# MIPSEL-MIPS2-NAN2008: Flags [ (0x10001404)

# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux -mcpu=mips1 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS1 %s
# MIPS1: Flags [ (0x1004)

 # RUN: llvm-mc -filetype=obj -triple mips-unknown-linux -mcpu=mips1 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS1-NAN2008 %s
# MIPS1-NAN2008: Flags [ (0x1404)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -target-abi o32 -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-O32 %s
# MIPS64EL-MIPS64-O32: Flags [ (0x60001104)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -target-abi o32 -mattr=+nan2008 -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-O32-NAN2008 %s
# MIPS64EL-MIPS64-O32-NAN2008: Flags [ (0x60001504)

# Default ABI for MIPS64 is N64 as opposed to GCC/GAS (N32)
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2 %s
# MIPS64EL-MIPS64R2: Flags [ (0x80000006)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64R2-NAN2008 %s
# MIPS64EL-MIPS64R2-NAN2008: Flags [ (0x80000406)

# Default ABI for MIPS64 is N64 as opposed to GCC/GAS (N32)
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64 %s
# MIPS64EL-MIPS64: Flags [ (0x60000006)

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64 -mattr=+nan2008 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPS64EL-MIPS64-NAN2008 %s
# MIPS64EL-MIPS64-NAN2008: Flags [ (0x60000406)

# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=octeon -target-abi n64 %s -o -| llvm-readobj -h | FileCheck --check-prefix=MIPSEL-OCTEON %s
# MIPSEL-OCTEON: Flags [ (0x808B0006)
