# Test 32-bit COMPARE IMMEDIATE AND BRANCH in cases where the sheer number of
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
#   0xffcc bytes, from MVIY instructions
#   conditional branch to main
# after0:
#   ...
#   conditional branch to main
# afterN:
#
# Each conditional branch sequence occupies 12 bytes if it uses a short
# branch and 16 if it uses a long one.  The ones before "main:" have to
# take the branch length into account, which is 6 for short branches,
# so the final (0x34 - 6) / 12 == 3 blocks can use short branches.
# The ones after "main:" do not, so the first 0x34 / 12 == 4 blocks
# can use short branches.  The conservative algorithm we use makes
# one of the forward branches unnecessarily long, as noted in the
# check output below.
#
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 50
# CHECK: jgl [[LABEL:\.L[^ ]*]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 51
# CHECK: jgl [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 52
# CHECK: jgl [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 53
# CHECK: jgl [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 54
# CHECK: jgl [[LABEL]]
# ...as mentioned above, the next one could be a CIJL instead...
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 55
# CHECK: jgl [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: cijl [[REG]], 56, [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: cijl [[REG]], 57, [[LABEL]]
# ...main goes here...
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: cijl [[REG]], 100, [[LABEL:\.L[^ ]*]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: cijl [[REG]], 101, [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: cijl [[REG]], 102, [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: cijl [[REG]], 103, [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 104
# CHECK: jgl [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 105
# CHECK: jgl [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 106
# CHECK: jgl [[LABEL]]
# CHECK: lb [[REG:%r[0-5]]], 0(%r3)
# CHECK: chi [[REG]], 107
# CHECK: jgl [[LABEL]]

from __future__ import print_function

branch_blocks = 8
main_size = 0xffcc

print('@global = global i32 0')

print('define void @f1(i8 *%base, i8 *%stop) {')
print('entry:')
print('  br label %before0')
print('')

for i in range(branch_blocks):
    next = 'before%d' % (i + 1) if i + 1 < branch_blocks else 'main'
    print('before%d:' % i)
    print('  %%bcur%d = load i8 , i8 *%%stop' % i)
    print('  %%bext%d = sext i8 %%bcur%d to i32' % (i, i))
    print('  %%btest%d = icmp slt i32 %%bext%d, %d' % (i, i, i + 50))
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
    print('  %%acur%d = load i8 , i8 *%%stop' % i)
    print('  %%aext%d = sext i8 %%acur%d to i32' % (i, i))
    print('  %%atest%d = icmp slt i32 %%aext%d, %d' % (i, i, i + 100))
    print('  br i1 %%atest%d, label %%main, label %%after%d' % (i, i))
    print('')
    print('after%d:' % i)

print('  %dummy = load volatile i32, i32 *@global')
print('  ret void')
print('}')
