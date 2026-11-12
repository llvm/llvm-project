# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu -show-encoding %s | \
# RUN:   FileCheck %s --check-prefix=CHECK-BE
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu -show-encoding %s | \
# RUN:   FileCheck %s --check-prefix=CHECK-LE

# Prefixed no-op (Power ISA v3.1, Section 3.3.1.2)
# Encoding: prefix 0x0700_0000 + suffix 0x0000_0000
pnop
# CHECK-BE: pnop  # encoding: [0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
# CHECK-LE: pnop  # encoding: [0x00,0x00,0x00,0x07,0x00,0x00,0x00,0x00]
