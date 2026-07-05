# RUN: not llvm-mc -triple i386-apple-darwin -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

# NOTE: apple-darwin portion of the triple is to enforce the convention choice
#       of what an assembler local symbol looks like (i.e., 'L' prefix.)

# CHECK: error: assembler local symbol 'Lbar' not defined
foo:
  jmp Lbar
