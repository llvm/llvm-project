// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping \
// RUN:        %S/Inputs/merge-same-func-bin1.c %S/Inputs/merge-same-func-bin1-2.c -o %t.bin1
// RUN: %t.bin1
// RUN: cp default.profraw %t.bin1.profraw 
//

// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping \
// RUN:        %S/Inputs/merge-same-func-bin2.c %S/Inputs/merge-same-func-bin1-2.c -o %t.bin2
// RUN: %t.bin2 || true
// RUN: cp default.profraw %t.bin2.profraw  
//---------------- merge the raw profiles ------------------------------------//
// RUN: llvm-profdata merge --object-aware-hashing=%t.bin2 %t.bin2.profraw \
// RUN:                     --object-aware-hashing=%t.bin1 %t.bin1.profraw \
// RUN:                     -o %t.profdata
//

// RUN: llvm-cov show -instr-profile=%t.profdata --object=%t.bin2 --object=%t.bin1 --merge-binary-coverage | FileCheck %s
//
// CHECK-LABEL: {{.*merge-same-func-bin1-2\.c}}:
// CHECK:     1|      1|int foo() { return 0; }
// CHECK:     2|       |
// CHECK:     3|      1|int bar() { return 0; }
// CHECK:     4|       |
// CHECK:     5|      1|int bun() { return 0; }
//
// CHECK-LABEL: {{.*merge-same-func-bin1\.c}}:
// CHECK:     1|       |extern int foo();
// CHECK:     2|       |extern int bar();
// CHECK:     3|       |extern int bun();
// CHECK:     4|       |
// CHECK:     5|       |
// CHECK:     6|      1|int main()
// CHECK:     7|      1|  return foo() + bar() + bun();
// CHECK:     8|      1|}
//
// CHECK-LABEL: {{.*merge-same-func-bin2\.c}}:
// CHECK:     1|      0|int baz() { return 0; }
// CHECK:     2|      1|int main() { return 1; }
