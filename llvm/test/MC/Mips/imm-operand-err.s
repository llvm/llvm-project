## Print an error if a non-immediate operand is used while an immediate is expected
# RUN: not llvm-mc -filetype=obj -triple=mips -o /dev/null %s 2>&1 | FileCheck %s --implicit-check-not=error:
# RUN: not llvm-mc -filetype=obj -triple=mips64 -o /dev/null %s 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: [[#@LINE+1]]:16: error: unsupported relocation type
  ori  $4, $4, start
  ori  $4, $4, (start - .)

# CHECK: [[#@LINE+1]]:18: error: unsupported relocation type
  addiu  $4, $4, start
  addiu  $4, $4, (start - .)

start:
