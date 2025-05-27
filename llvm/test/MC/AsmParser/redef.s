# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

l:

.data
.set x, 0
.long x
x = .-.data
.long x
.set x,.-.data
# CHECK: [[#@LINE-1]]:8: error: invalid reassignment of non-absolute variable 'x'
## TODO This should be allowed
.long x

.globl l_v
.set l_v, l
.globl l_v
.set l_v, l
