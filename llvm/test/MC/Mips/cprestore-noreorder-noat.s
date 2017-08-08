# RUN: not llvm-mc %s -arch=mips -mcpu=mips32 --position-independent -filetype=obj \
# RUN:   -o /dev/null 2>&1 | FileCheck %s -check-prefix=O32

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 \
# RUN:   -filetype=obj -o /dev/null 2>&1 | FileCheck %s -allow-empty -check-prefix=N32

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi=n32 \
# RUN:   -filetype=obj -o /dev/null 2>&1 | FileCheck %s -allow-empty -check-prefix=N64

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi=n32 \
# RUN:   -filetype=obj -o - | llvm-objdump -d -r - | \
# RUN:   FileCheck %s -check-prefix=NO-STORE

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -filetype=obj -o - | \
# RUN:   llvm-objdump -d -r - | FileCheck %s -check-prefix=NO-STORE

  .text
  .ent foo
foo:
  .frame  $sp, 0, $ra
  .set noreorder
  .set noat

  .cpload $25
  .cprestore 8
# O32-NOT: error: pseudo-instruction requires $at, which is not available
# N32-NOT: error: pseudo-instruction requires $at, which is not available
# N64-NOT: error: pseudo-instruction requires $at, which is not available
# NO-STORE-NOT: sw  $gp, 8($sp)

  jal $25
  jal $4, $25
  jal foo

  .end foo

  .ent bar
bar:
  .frame  $sp, 0, $ra
  .set noreorder
  .set noat

  .cpload $25
  .cprestore 65536
# O32: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
# N32-NOT: error: pseudo-instruction requires $at, which is not available
# N64-NOT: error: pseudo-instruction requires $at, which is not available
# NO-STORE-NOT: sw $gp,

  jal $25
  jal $4, $25
  jal bar

  .end bar
