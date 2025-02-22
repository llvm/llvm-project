# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-xrivosvizip < %s 2>&1 | \
# RUN:   FileCheck %s

# Disallowed source/dest overlap cases
# CHECK:  error: the destination vector register group cannot overlap the source vector register group
rv.vzipeven.vv v2, v2, v3
# CHECK:  error: the destination vector register group cannot overlap the source vector register group
rv.vzipeven.vv v3, v2, v3
# CHECK: error: the destination vector register group cannot overlap the mask register
rv.vzipeven.vv v0, v2, v3, v0.t
