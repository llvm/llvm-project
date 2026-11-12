# RUN: not llvm-mc -triple i386-unknown-unknown %s -I %p | FileCheck %s
# RUN: not llvm-mc -triple i386-unknown-unknown %s -I %p 2>&1 > /dev/null| FileCheck %s --check-prefix=CHECK-ERROR

.data
.incbin "incbin\137abcd"  # "\137" is underscore "_"

# CHECK: .ascii	 "abcd\n"

.data
.incbin "incbin\137abcd", 1

# CHECK: .ascii	 "bcd\n"

.data
.incbin "incbin\137abcd", 1, 2

# CHECK: .ascii	 "bc"

.data
.incbin "incbin\137abcd",, 2

# CHECK: .ascii	 "ab"

.data
.incbin incbin\137abcd

# CHECK-ERROR: error: expected string in '.incbin' directive

.data
.incbin "incbin\137abcd" 1
# CHECK-ERROR: :[[#@LINE-1]]:26: error: expected newline

.data
.incbin "incbin\137abcd", 1 2
# CHECK-ERROR: :[[#@LINE-1]]:29: error: expected newline

.data
.incbin "incbin\137abcd", -1

# CHECK-ERROR: error: skip is negative

.data
.incbin "incbin\137abcd",, -1

# CHECK-ERROR: warning: negative count has no effect
