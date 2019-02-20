# Test cases where MVC is used for spill slots that end up being out of range.
# RUN: python %s | llc -mtriple=s390x-linux-gnu | FileCheck %s

# There are 8 usable call-saved GPRs, two of which are needed for the base
# registers.  The first 160 bytes of the frame are needed for the ABI
# call frame, and a further 8 bytes are needed for the emergency spill slot.
# That means we will have at least one out-of-range slot if:
#
#    count == (4096 - 168) / 8 + 6 + 1 == 498
#
# Add in some extra room and check both %r15+4096 (the first out-of-range slot)
# and %r15+4104.
#
# CHECK: f1:
# CHECK: lay [[REG:%r[0-5]]], 4096(%r15)
# CHECK: mvc 0(8,[[REG]]), {{[0-9]+}}({{%r[0-9]+}})
# CHECK: brasl %r14, foo@PLT
# CHECK: lay [[REG:%r[0-5]]], 4096(%r15)
# CHECK: mvc {{[0-9]+}}(8,{{%r[0-9]+}}), 8([[REG]])
# CHECK: br %r14

from __future__ import print_function

count = 500

print('declare void @foo()')
print('')
print('define void @f1(i64 *%base0, i64 *%base1) {')

for i in range(count):
    print('  %%ptr%d = getelementptr i64, i64 *%%base%d, i64 %d' % (i, i % 2, i / 2))
    print('  %%val%d = load i64 , i64 *%%ptr%d' % (i, i))
    print('')

print('  call void @foo()')
print('')

for i in range(count):
    print('  store i64 %%val%d, i64 *%%ptr%d' % (i, i))

print('')
print('  ret void')
print('}')
