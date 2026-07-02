# RUN: llvm-mc -triple=x86_64-unknown-linux-gnu -mattr=+lvi-cfi -x86-experimental-lvi-inline-asm-hardening -show-encoding %s | FileCheck %s --check-prefix=X64
# RUN: llvm-mc -triple=i386-unknown-linux-gnu -mattr=+lvi-cfi -x86-experimental-lvi-inline-asm-hardening -show-encoding %s | FileCheck %s --check-prefix=X32

ret

# X64: shlq $0, (%rsp)
# X64-NEXT: lfence
# X64-NEXT: retq

# X32: shll $0, (%esp)
# X32-NEXT: lfence
# X32-NEXT: retl
