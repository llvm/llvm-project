# RUN: llvm-mc -triple x86_64-unknown-darwin %s | FileCheck %s

# CHECK: .tbss _a$tlv$init, 4
# CHECK: .tbss _b$tlv$init, 4, 3

.tbss _a$tlv$init, 4
.tbss _b$tlv$init, 4, 3
