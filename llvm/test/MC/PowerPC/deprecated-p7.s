# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu -mcpu=pwr7 -show-encoding < %s 2>&1 | FileCheck %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu -mcpu=pwr7 -show-encoding < %s 2>&1 | FileCheck %s
# RUN: llvm-mc -triple powerpc-unknown-linux-gnu -mcpu=601 -show-encoding < %s 2>&1 | FileCheck -check-prefix=CHECK-OLD %s

         mftb 3
# CHECK-NOT: warning: deprecated
# CHECK: mfspr 3, 268

# CHECK-OLD-NOT: warning: deprecated
# CHECK-OLD: mftb 3

# FIXME: Test dst and friends once we can parse them.

