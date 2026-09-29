# RUN: not llvm-mc -triple i386-unknown-unknown %s | FileCheck %s
# RUN: not llvm-mc -triple i386-unknown-unknown %s 2>&1 > /dev/null| FileCheck %s --check-prefix=CHECK-ERROR

# CHECK: TEST0:
# CHECK: .zero 1
TEST0:
        .ds.b 1

# CHECK: TEST1:
# CHECK: .zero 2
# CHECK: .zero 2
# CHECK: .zero 2
TEST1:
        .ds 3

# CHECK: TEST2:
TEST2:
        .ds.w 0

# CHECK: TEST3:
# CHECK: .zero 4
# CHECK: .zero 4
TEST3:
        .ds.l 2

# CHECK: TEST4:
# CHECK: .zero 8
# CHECK: .zero 8
# CHECK: .zero 8
# CHECK: .zero 8
TEST4:
        .ds.d 4

# CHECK: TEST5:
# CHECK: .zero 12
# CHECK: .zero 12
TEST5:
        .ds.p 2

# CHECK: TEST6:
# CHECK: .zero 4
# CHECK: .zero 4
# CHECK: .zero 4
TEST6:
        .ds.s 3

# CHECK: TEST7:
# CHECK: .zero 12
TEST7:
        .ds.x 1

# CHECK-ERROR: warning: '.ds' directive with negative repeat count has no effect
TEST8:
       .ds -1

# CHECK-ERROR: :[[#@LINE+2]]:9: error: expected newline
TEST9:
  .ds 1 2
