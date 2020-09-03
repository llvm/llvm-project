## Print an error if a non-immediate operand is used while an immediate is expected
# RUN: not llvm-mc -filetype=obj -triple=mips -o /dev/null %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple=mips64 -o /dev/null %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:16: error: expected an immediate
  ori  $4, $4, start
# CHECK: [[#@LINE+1]]:17: error: expected an immediate
  ori  $4, $4, (start - .)

# CHECK: [[#@LINE+1]]:18: error: expected an immediate
  addiu  $4, $4, start
# CHECK: [[#@LINE+1]]:19: error: expected an immediate
  addiu  $4, $4, (start - .)

start:
