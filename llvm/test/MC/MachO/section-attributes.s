// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o %t
// RUN: macho-dump %t | FileCheck %s

// CHECK: # Section 1
// CHECK: ('flags', 0x0)
.section __TEXT,__objc_opt_ro
.long 0
