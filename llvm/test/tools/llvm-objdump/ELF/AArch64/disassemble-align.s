# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
# RUN: llvm-objdump -d %t | tr '\t' '|' | FileCheck --match-full-lines --strict-whitespace %s

## Use '|' to show where the tabs line up.
#       CHECK:0000000000000000 <$x.0>:
#  CHECK-NEXT:       0: 62 10 00 91  |add|x2, x3, #4
# CHECK-EMPTY:
#  CHECK-NEXT:0000000000000004 <$d.1>:
#  CHECK-NEXT:       4:|ff ff 00 00|.word|0x0000ffff

  add x2, x3, #4
  .word 0xffff
