# RUN: not llvm-mc -triple powerpc64-unknown-linux-gnu %s 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-ERR

# Valid wrteei encodings are already covered by
# MC/PowerPC/ppc64-encoding-bookIII.s; this test only checks that invalid
# operands are rejected.

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
