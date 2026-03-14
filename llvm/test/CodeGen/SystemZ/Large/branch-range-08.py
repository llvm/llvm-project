# Test 64-bit BRANCH RELATIVE ON COUNT in cases where some branches are out
# of range.
# RUN: %python %s | llc -mtriple=s390x-linux-gnu | FileCheck %s

# Construct:
#
# loopN:
#   load of countN
#   ...
# loop0:
#   0xffd8 bytes, from MVIY instructions
#   conditional branch to main
# after0:
#   ...
#   decrement of countN
#   conditional branch to loopN
# afterN:
#
# Each load occupies 6 bytes.  Each decrement and branch occupies 4
# bytes if BRCTG can be used, otherwise it occupies 10 bytes (AGHI + BRCL).
# This means that loop 5 contains 4 * 6 + 0xffd8 + 4 * 4 == 0x10000 bytes
# and is therefore (just) in range.  Loop 6 is out of range.
#
# CHECK: brctg {{%r[0-9]+}}
# CHECK: brctg {{%r[0-9]+}}
# CHECK: brctg {{%r[0-9]+}}
# CHECK: brctg {{%r[0-9]+}}
# CHECK: brctg {{%r[0-9]+}}
# CHECK: aghi {{%r[0-9]+}}, -1
# CHECK: jglh
# CHECK: aghi {{%r[0-9]+}}, -1
# CHECK: jglh
# CHECK: aghi {{%r[0-9]+}}, -1
# CHECK: jglh

from __future__ import print_function

branch_blocks = 8
main_size = 0xFFD8

print("define void @f1(i8 *%base, i64 *%counts) {")
print("entry:")

for i in range(branch_blocks - 1, -1, -1):
    print("  %%countptr%d = getelementptr i64, i64 *%%counts, i64 %d" % (i, i))
    print("  %%initcount%d = load i64 , i64 *%%countptr%d" % (i, i))
    print("  br label %%loop%d" % i)

    print("loop%d:" % i)
    block1 = "entry" if i == branch_blocks - 1 else "loop%d" % (i + 1)
    block2 = "loop0" if i == 0 else "after%d" % (i - 1)
    print(
        (
            "  %%count%d = phi i64 [ %%initcount%d, %%%s ],"
            " [ %%nextcount%d, %%%s ]" % (i, i, block1, i, block2)
        )
    )

a, b = 1, 1
for i in range(0, main_size, 6):
    a, b = b, a + b
    offset = 4096 + b % 500000
    value = a % 256
    print("  %%ptr%d = getelementptr i8, i8 *%%base, i64 %d" % (i, offset))
    print("  store volatile i8 %d, i8 *%%ptr%d" % (value, i))

for i in range(branch_blocks):
    print("  %%nextcount%d = add i64 %%count%d, -1" % (i, i))
    print("  %%test%d = icmp ne i64 %%nextcount%d, 0" % (i, i))
    print("  br i1 %%test%d, label %%loop%d, label %%after%d" % (i, i, i))
    print("")
    print("after%d:" % i)

print("  ret void")
print("}")
