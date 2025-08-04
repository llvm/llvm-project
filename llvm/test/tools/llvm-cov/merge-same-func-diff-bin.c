// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping \
// RUN:        %S/Inputs/merge-same-func-bin1.c %S/Inputs/merge-same-func-bin1-2.c -o %t.hex
// RUN: %t.hex
// RUN: cp default.profraw %t.hex.profraw 
//
//---------------- build & run the “x86” variant ------------------------------//
// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping \
// RUN:        %S/Inputs/merge-same-func-bin2.c %S/Inputs/merge-same-func-bin1-2.c -o %t.x86
// RUN: %t.x86 || true
// RUN: cp default.profraw %t.x86.profraw  
//---------------- merge the raw profiles ------------------------------------//
// RUN: llvm-profdata merge --object-aware-hashing=%t.x86 %t.x86.profraw \
// RUN:                     --object-aware-hashing=%t.hex %t.hex.profraw \
// RUN:                     -o %t.profdata
//
//---------------- show unified coverage & check -----------------------------//
// RUN: llvm-cov show -instr-profile=%t.profdata --object=%t.x86 --object=%t.hex --merge-binary-coverage | FileCheck %s
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
