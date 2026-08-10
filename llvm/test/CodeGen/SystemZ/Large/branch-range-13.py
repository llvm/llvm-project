# Test that inline assembly get the right size value so that a branch across
# a block containing them gets relaxed.

# RUN: %python %s | llc -mtriple=s390x-linux-gnu -mcpu=z196 -enable-post-misched=false \
# RUN:    | FileCheck %s

# Construct:
#
# entry:
#   branch to block
#
# block:
#   sequence of call asm
#   unconditional branch to block
#
# exit:
#   ret void

# CHECK-LABEL: f1
# CHECK: jg
# CHECK-NEXT: .Lfunc_end0:

from __future__ import print_function

num = 11000

print("define void @f1() {")
print("entry:")
print("  br label %block")
print("")
print("block:")

for i in range(num):
    print(
        '  tail call i64 asm "lang\\09$0,$2,$1\\0A", "=d,=*Q,d,*Q"(i32* elementtype(i32) undef, i32 undef, i32* elementtype(i32) undef)'
    )

print("  br label %block")

print("")
print("exit:")
print("  ret void")
print("}")
