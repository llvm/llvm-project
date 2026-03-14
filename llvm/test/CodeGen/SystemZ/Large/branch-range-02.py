# Test normal conditional branches in cases where block alignments cause
# some branches to be out of range.
# RUN: %python %s | llc -mtriple=s390x-linux-gnu -align-all-blocks=8 | FileCheck %s

# Construct:
#
# b0:
#   conditional branch to end
#   ...
# b<N>:
#   conditional branch to end
# b<N+1>:
#   conditional branch to b0
#   ...
# b<2*N>:
#   conditional branch to b0
# end:
#
# with N == 256 + 4.  The -align-all-blocks=8 option ensures that all blocks
# are 256 bytes in size.  The first 4 blocks and the last 4 blocks are then
# out of range.
#
# CHECK: c %r4, 0(%r3)
# CHECK: jge [[LABEL:\.L[^ ]*]]
# CHECK: c %r4, 4(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 8(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 12(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 16(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 20(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 24(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 28(%r3)
# CHECK: je [[LABEL]]
# ...lots of other blocks...
# CHECK: c %r4, 1004(%r3)
# CHECK: je [[LABEL:\.L[^ ]*]]
# CHECK: c %r4, 1008(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 1012(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 1016(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 1020(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 1024(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 1028(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 1032(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 1036(%r3)
# CHECK: jge [[LABEL]]

from __future__ import print_function

blocks = 256 + 4

print("define void @f1(i8 *%base, i32 *%stop, i32 %limit) {")
print("entry:")
print("  br label %b0")
print("")

a, b = 1, 1
for i in range(blocks):
    a, b = b, a + b
    value = a % 256
    next = "b%d" % (i + 1) if i + 1 < blocks else "end"
    other = "end" if 2 * i < blocks else "b0"
    print("b%d:" % i)
    print("  store volatile i8 %d, i8 *%%base" % value)
    print("  %%astop%d = getelementptr i32, i32 *%%stop, i64 %d" % (i, i))
    print("  %%acur%d = load i32 , i32 *%%astop%d" % (i, i))
    print("  %%atest%d = icmp eq i32 %%limit, %%acur%d" % (i, i))
    print("  br i1 %%atest%d, label %%%s, label %%%s" % (i, other, next))

print("")
print("%s:" % next)
print("  ret void")
print("}")
