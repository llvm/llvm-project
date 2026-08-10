# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s

gcsrrd $a0, 16384
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [0, 16383]

gcsrrd $a0, -1
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [0, 16383]

gcsrwr $a0, 16384
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [0, 16383]

gcsrwr $a0, -1
# CHECK: :[[#@LINE-1]]:13: error: immediate must be an integer in the range [0, 16383]

gcsrxchg $a0, $a1, 16384
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 16383]

gcsrxchg $a0, $a1, -1
# CHECK: :[[#@LINE-1]]:20: error: immediate must be an integer in the range [0, 16383]

gcsrxchg $a0, $ra, 1
# CHECK: :[[#@LINE-1]]:16: error: must not be $r0 or $r1

gcsrxchg $a0, $zero, 1
# CHECK: :[[#@LINE-1]]:16: error: must not be $r0 or $r1

hvcl 32768
# CHECK: :[[#@LINE-1]]:6: error: immediate must be an integer in the range [0, 32767]

hvcl -1
# CHECK: :[[#@LINE-1]]:6: error: immediate must be an integer in the range [0, 32767]
