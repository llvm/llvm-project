// Check that preprocessing respects macro definitions.

// RUN: llvm-tblgen %s -E -DFOO | FileCheck %s -check-prefix=CHECK-FOO
// RUN: llvm-tblgen %s -E | FileCheck %s

// CHECK-FOO: yes
// CHECK: no

#ifdef FOO
yes
#else
no
#endif
