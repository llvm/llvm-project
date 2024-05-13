# Test cases where we spill from one frame index to another, both of which
# are out of range of MVC, and both of which need emergency spill slots.
# RUN: %python %s | llc -mtriple=s390x-linux-gnu | FileCheck %s

# CHECK: f1:
# CHECK: %fallthru
# CHECK-DAG: stg [[REG1:%r[0-9]+]], 8168(%r15)
# CHECK-DAG: stg [[REG2:%r[0-9]+]], 8176(%r15)
# CHECK-DAG: lay [[REG3:%r[0-9]+]], 8192(%r15)
# CHECK-DAG: lay [[REG4:%r[0-9]+]], 4096(%r15)
# CHECK: mvc 0(8,[[REG3]]), 4088([[REG4]])
# CHECK-DAG: lg [[REG1]], 8168(%r15)
# CHECK-DAG: lg [[REG2]], 8176(%r15)
# CHECK: %skip
# CHECK: br %r14

# Arrange for %foo's spill slot to be at 8184(%r15) and the alloca area to be at
# 8192(%r15).  The two emergency spill slots live below that, so this requires
# the first 8168 bytes to be used for the call.  160 of these bytes are
# allocated for the ABI frame.  There are also 5 argument registers, one of
# which is used as a base pointer.

from __future__ import print_function

args = int((8168 - 160) / 8 + (5 - 1))

print("declare i64 *@foo(i64 *%s)" % (", i64" * args))
print("declare void @bar(i64 *)")
print("")
print("define i64 @f1(i64 %foo) {")
print("entry:")

# Make the allocation big, so that it goes at the top of the frame.
print("  %array = alloca [1000 x i64]")
print("  %area = getelementptr [1000 x i64], [1000 x i64] *%array, i64 0, i64 0")
print("  %%base = call i64 *@foo(i64 *%%area%s)" % (", i64 0" * args))
print("")

# Make sure all GPRs are used.  One is needed for the stack pointer and
# another for %base, so we need 14 live values.
count = 14
for i in range(count):
    print("  %%ptr%d = getelementptr i64, i64 *%%base, i64 %d" % (i, i / 2))
    print("  %%val%d = load volatile i64 , i64 *%%ptr%d" % (i, i))
    print("")

# Encourage the register allocator to give preference to these %vals
# by using them several times.
for j in range(4):
    for i in range(count):
        print("  store volatile i64 %%val%d, i64 *%%ptr%d" % (i, i))
    print("")

# Copy the incoming argument, which we expect to be spilled, to the frame
# index for the alloca area.  Also throw in a volatile store, so that this
# block cannot be reordered with the surrounding code.
print("  %cond = icmp eq i64 %val0, %val1")
print("  br i1 %cond, label %skip, label %fallthru")
print("")
print("fallthru:")
print("  store i64 %foo, i64 *%area")
print("  store volatile i64 %val0, i64 *%ptr0")
print("  br label %skip")
print("")
print("skip:")

# Use each %val a few more times to emphasise the point, and to make sure
# that they are live across the store of %foo.
for j in range(4):
    for i in range(count):
        print("  store volatile i64 %%val%d, i64 *%%ptr%d" % (i, i))
    print("")

print("  call void @bar(i64 *%area)")
print("  ret i64 0")
print("}")
