# RUN: not llvm-mc -triple s390x-unknown-unknown -filetype=obj -mcpu=zEC12 \
# RUN:   -o /dev/null %s 2>&1 | FileCheck %s

	.text

# Test fixup ranges, which are encoded as half-words.

# 12-bit
# CHECK:      error: operand out of range (4096 not between -4096 and 4094)
# CHECK-NEXT:        bprp 0, .Lab1, 0
# CHECK-NEXT:        ^
# CHECK-NEXT: error: operand out of range (-4098 not between -4096 and 4094)
# CHECK-NEXT:        bprp 0, .Lab0, 0
# CHECK-NEXT:        ^
        bprp 0, .Lab1, 0
.Lab0:
        bprp 0, .Lab1, 0
        .space 4084
.Lab1:
	nopr
        bprp 0, .Lab0, 0
        bprp 0, .Lab0, 0

# 24-bit
# CHECK-NEXT: error: operand out of range (16777220 not between -16777216 and 16777214)
# CHECK-NEXT:        bprp 0, 0, .Lab3
# CHECK-NEXT:        ^
# CHECK-NEXT: error: operand out of range (-16777222 not between -16777216 and 16777214)
# CHECK-NEXT:        bprp 0, 0, .Lab2
# CHECK-NEXT:        ^
        bprp 0, 0, .Lab3
.Lab2:
        bprp 0, 0, .Lab3
        .space 16777208
.Lab3:
	nopr
        bprp 0, 0, .Lab2
        bprp 0, 0, .Lab2

# 16-bit
# CHECK-NEXT: error: operand out of range (65540 not between -65536 and 65534)
# CHECK-NEXT:        cij %r1, 0, 0, .Lab5
# CHECK-NEXT:        ^
# CHECK-NEXT: error: operand out of range (-65542 not between -65536 and 65534)
# CHECK-NEXT:        cij %r1, 0, 0, .Lab4
# CHECK-NEXT:        ^
        cij %r1, 0, 0, .Lab5
.Lab4:
        cij %r1, 0, 0, .Lab5
        .space 65528
.Lab5:
	nopr
        cij %r1, 0, 0, .Lab4
        cij %r1, 0, 0, .Lab4

# 32-bit
# Unfortunately there is no support for offsets greater than 32 bits, so we have
# to for now assume they are in range.

# CHECK-NOT: error
