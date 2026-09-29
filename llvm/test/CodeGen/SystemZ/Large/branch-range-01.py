# Test normal conditional branches in cases where the sheer number of
# instructions causes some branches to be out of range.
# RUN: %python %s | llc -mtriple=s390x-linux-gnu | FileCheck %s

# Construct:
#
# before0:
#   conditional branch to after0
#   ...
# beforeN:
#   conditional branch to after0
# main:
#   0xffd8 bytes, from MVIY instructions
#   conditional branch to main
# after0:
#   ...
#   conditional branch to main
# afterN:
#
# Each conditional branch sequence occupies 8 bytes if it uses a short branch
# and 10 if it uses a long one.  The ones before "main:" have to take the branch
# length into account -- which is 4 bytes for short branches -- so the final
# (0x28 - 4) / 8 == 4 blocks can use short branches.  The ones after "main:"
# do not, so the first 0x28 / 8 == 5 can use short branches.  However,
# the conservative algorithm we use makes one branch unnecessarily long
# on each side.
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
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 20(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 24(%r3)
# CHECK: j{{g?}}e [[LABEL]]
# CHECK: c %r4, 28(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 32(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 36(%r3)
# CHECK: je [[LABEL]]
# ...main goes here...
# CHECK: c %r4, 100(%r3)
# CHECK: je [[LABEL:\.L[^ ]*]]
# CHECK: c %r4, 104(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 108(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 112(%r3)
# CHECK: je [[LABEL]]
# CHECK: c %r4, 116(%r3)
# CHECK: j{{g?}}e [[LABEL]]
# CHECK: c %r4, 120(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 124(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 128(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 132(%r3)
# CHECK: jge [[LABEL]]
# CHECK: c %r4, 136(%r3)
# CHECK: jge [[LABEL]]

from __future__ import print_function

branch_blocks = 10
main_size = 0xFFD8

print("@global = global i32 0")

print("define void @f1(i8 *%base, i32 *%stop, i32 %limit) {")
print("entry:")
print("  br label %before0")
print("")

for i in range(branch_blocks):
    next = "before%d" % (i + 1) if i + 1 < branch_blocks else "main"
    print("before%d:" % i)
    print("  %%bstop%d = getelementptr i32, i32 *%%stop, i64 %d" % (i, i))
    print("  %%bcur%d = load i32 , i32 *%%bstop%d" % (i, i))
    print("  %%btest%d = icmp eq i32 %%limit, %%bcur%d" % (i, i))
    print("  br i1 %%btest%d, label %%after0, label %%%s" % (i, next))
    print("")

print("%s:" % next)
a, b = 1, 1
for i in range(0, main_size, 6):
    a, b = b, a + b
    offset = 4096 + b % 500000
    value = a % 256
    print("  %%ptr%d = getelementptr i8, i8 *%%base, i64 %d" % (i, offset))
    print("  store volatile i8 %d, i8 *%%ptr%d" % (value, i))

for i in range(branch_blocks):
    print("  %%astop%d = getelementptr i32, i32 *%%stop, i64 %d" % (i, i + 25))
    print("  %%acur%d = load i32 , i32 *%%astop%d" % (i, i))
    print("  %%atest%d = icmp eq i32 %%limit, %%acur%d" % (i, i))
    print("  br i1 %%atest%d, label %%main, label %%after%d" % (i, i))
    print("")
    print("after%d:" % i)

print("  %dummy = load volatile i32, i32 *@global")
print("  ret void")
print("}")
