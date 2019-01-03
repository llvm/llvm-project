# Test 64-bit COMPARE LOGICAL IMMEDIATE AND BRANCH in cases where the sheer
# number of instructions causes some branches to be out of range.
# RUN: python %s | llc -mtriple=s390x-linux-gnu | FileCheck %s

# Construct:
#
# before0:
#   conditional branch to after0
#   ...
# beforeN:
#   conditional branch to after0
# main:
#   0xffb4 bytes, from MVIY instructions
#   conditional branch to main
# after0:
#   ...
#   conditional branch to main
# afterN:
#
# Each conditional branch sequence occupies 18 bytes if it uses a short
# branch and 24 if it uses a long one.  The ones before "main:" have to
# take the branch length into account, which is 6 for short branches,
# so the final (0x4c - 6) / 18 == 3 blocks can use short branches.
# The ones after "main:" do not, so the first 0x4c / 18 == 4 blocks
# can use short branches.  The conservative algorithm we use makes
# one of the forward branches unnecessarily long, as noted in the
# check output below.
#
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 50
# CHECK: jgl [[LABEL:\.L[^ ]*]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 51
# CHECK: jgl [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 52
# CHECK: jgl [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 53
# CHECK: jgl [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 54
# CHECK: jgl [[LABEL]]
# ...as mentioned above, the next one could be a CLGIJL instead...
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 55
# CHECK: jgl [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgijl [[REG]], 56, [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgijl [[REG]], 57, [[LABEL]]
# ...main goes here...
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgijl [[REG]], 100, [[LABEL:\.L[^ ]*]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgijl [[REG]], 101, [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgijl [[REG]], 102, [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgijl [[REG]], 103, [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 104
# CHECK: jgl [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 105
# CHECK: jgl [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 106
# CHECK: jgl [[LABEL]]
# CHECK: lg [[REG:%r[0-5]]], 0(%r3)
# CHECK: sg [[REG]], 0(%r4)
# CHECK: clgfi [[REG]], 107
# CHECK: jgl [[LABEL]]

from __future__ import print_function

branch_blocks = 8
main_size = 0xffb4

print('@global = global i32 0')

print('define void @f1(i8 *%base, i64 *%stopa, i64 *%stopb) {')
print('entry:')
print('  br label %before0')
print('')

for i in range(branch_blocks):
    next = 'before%d' % (i + 1) if i + 1 < branch_blocks else 'main'
    print('before%d:' % i)
    print('  %%bcur%da = load i64 , i64 *%%stopa' % i)
    print('  %%bcur%db = load i64 , i64 *%%stopb' % i)
    print('  %%bsub%d = sub i64 %%bcur%da, %%bcur%db' % (i, i, i))
    print('  %%btest%d = icmp ult i64 %%bsub%d, %d' % (i, i, i + 50))
    print('  br i1 %%btest%d, label %%after0, label %%%s' % (i, next))
    print('')

print('%s:' % next)
a, b = 1, 1
for i in range(0, main_size, 6):
    a, b = b, a + b
    offset = 4096 + b % 500000
    value = a % 256
    print('  %%ptr%d = getelementptr i8, i8 *%%base, i64 %d' % (i, offset))
    print('  store volatile i8 %d, i8 *%%ptr%d' % (value, i))

for i in range(branch_blocks):
    print('  %%acur%da = load i64 , i64 *%%stopa' % i)
    print('  %%acur%db = load i64 , i64 *%%stopb' % i)
    print('  %%asub%d = sub i64 %%acur%da, %%acur%db' % (i, i, i))
    print('  %%atest%d = icmp ult i64 %%asub%d, %d' % (i, i, i + 100))
    print('  br i1 %%atest%d, label %%main, label %%after%d' % (i, i))
    print('')
    print('after%d:' % i)

print('  %dummy = load volatile i32, i32 *@global')
print('  ret void')
print('}')
