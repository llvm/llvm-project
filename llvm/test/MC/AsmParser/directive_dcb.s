# RUN: not llvm-mc -triple i386-unknown-unknown %s | FileCheck %s
# RUN: not llvm-mc -triple i386-unknown-unknown %s 2>&1 > /dev/null| FileCheck %s --check-prefix=CHECK-ERROR

# CHECK: TEST0:
# CHECK: .byte 1
# CHECK: .byte 1
TEST0:
        .dcb.b 2, 1

# CHECK: TEST1:
# CHECK: .short 3
TEST1:
        .dcb 1, 3

# CHECK: TEST2:
# CHECK: .short 3
# CHECK: .short 3
TEST2:
        .dcb.w 2, 3

# CHECK: TEST3:
# CHECK: .long 8
# CHECK: .long 8
# CHECK: .long 8
TEST3:
        .dcb.l 3, 8

# CHECK: TEST5
# CHECK: .long	1067412619
# CHECK: .long	1067412619
# CHECK: .long	1067412619
# CHECK: .long	1067412619
TEST5:
        .dcb.s 4, 1.2455

# CHECK: TEST6
# CHECK: .quad	4597526701198935065
# CHECK: .quad	4597526701198935065
# CHECK: .quad	4597526701198935065
# CHECK: .quad	4597526701198935065
# CHECK: .quad	4597526701198935065
TEST6:
        .dcb.d 5, .232

# CHECK-ERROR: error: .dcb.x not currently supported for this target
TEST7:
        .dcb.x 3, 1.2e3

# CHECK-ERROR: warning: '.dcb' directive with negative repeat count has no effect
TEST8:
       .dcb -1, 2

# CHECK-ERROR: error: unexpected token in '.dcb' directive
TEST9:
       .dcb 1 2

# CHECK-ERROR: error: unexpected token in '.dcb' directive
TEST10:
       .dcb 1, 2 3
