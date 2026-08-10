# RUN: llvm-mc -triple=i386 %s | FileCheck %s
# RUN: not llvm-mc -triple=i386 --defsym ERR=1 %s 2>&1 > /dev/null | FileCheck %s --check-prefix=ERR

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

.ifdef ERR
# ERR: :[[#@LINE+1]]:8: error: .dcb.x not currently supported for this target
.dcb.x 3, 1.2e3

# ERR: :[[#@LINE+1]]:6: warning: '.dcb' directive with negative repeat count has no effect
.dcb -1, 2

# ERR: :[[#@LINE+1]]:8: error: expected comma
.dcb 1 2

# ERR: :[[#@LINE+1]]:11: error: expected newline
.dcb 1, 2 3
.endif
