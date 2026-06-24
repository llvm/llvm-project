// Check that parsing the preprocessor output creates exactly the same
// records as parsing the original files directly. Use Target.td as a
// reasonably complex sample input.

// RUN: llvm-tblgen %p/../../include/llvm/Target/Target.td -I %p/../../include -o %t -E
// RUN: llvm-tblgen %p/../../include/llvm/Target/Target.td -I %p/../../include -o %t1
// RUN: llvm-tblgen %t -o %t2
// RUN: cmp %t1 %t2

// Check that preprocessing this file respects macro definitions.

// RUN: llvm-tblgen %s -E -DFOO | FileCheck %s -check-prefix=CHECK-FOO
// RUN: llvm-tblgen %s -E | FileCheck %s

// CHECK-FOO: yes
// CHECK: no

#ifdef FOO
yes
#else
no
#endif

// RUN: echo '/*' | not llvm-tblgen -E 2>&1 | grep "error: unterminated comment"
// RUN: echo 'include "foo"' | not llvm-tblgen -E 2>&1 | grep "error: could not find include file"
// RUN: echo '!foo' | not llvm-tblgen -E 2>&1 | grep "error: unknown operator"
// RUN: echo '#else' | not llvm-tblgen -E 2>&1 | grep "error: #else without #ifdef or #ifndef"
