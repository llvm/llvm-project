; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:                 -mcpu=pwr4 -mattr=-altivec < %s 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: section's multiply symbols policy does not match

@a = global i32 3, section "ab", align 4
@ab = global i32 5, align 4
