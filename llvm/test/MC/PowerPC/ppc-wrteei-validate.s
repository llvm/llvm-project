# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu -show-encoding %s 2>&1 | \
# RUN:   FileCheck %s

# RUN: not llvm-mc -triple powerpc64-unknown-linux-gnu %s --defsym=ERR=1 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-ERR

# Valid wrteei operands (0 and 1)
wrteei 0
# CHECK: wrteei 0
wrteei 1
# CHECK: wrteei 1

.ifdef ERR
# Invalid: register names should be rejected as immediate operands
wrteei f0
# CHECK-ERR: [[@LINE-1]]:{{[0-9]+}}: error:
wrteei r0
# CHECK-ERR: [[@LINE-1]]:{{[0-9]+}}: error:
wrteei cr0
# CHECK-ERR: [[@LINE-1]]:{{[0-9]+}}: error:
wrteei v0
# CHECK-ERR: [[@LINE-1]]:{{[0-9]+}}: error:
# Invalid: out of range
wrteei 2
# CHECK-ERR: [[@LINE-1]]:{{[0-9]+}}: error:
wrteei -1
# CHECK-ERR: [[@LINE-1]]:{{[0-9]+}}: error:
.endif
