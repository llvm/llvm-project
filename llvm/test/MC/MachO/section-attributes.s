// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o %t
// RUN: llvm-readobj -S --sd %t | FileCheck %s

.section __TEXT,__objc_opt_ro
.long 0

// CHECK: Section {
// CHECK:   Index: 1
// CHECK:   Attributes [ (0x0)
// CHECK:   ]
