// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping \
// RUN:        %S/Inputs/myprogram.c %S/Inputs/fb.c -o %t.hex
// RUN: env LLVM_PROFILE_FILE=%t.hex.profraw 
// RUN: %t.hex
//
//---------------- build & run the “x86” variant ------------------------------//
// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping \
// RUN:        %S/Inputs/mypg.c %S/Inputs/fb.c -o %t.x86
// RUN: env LLVM_PROFILE_FILE=%t.x86.profraw 
// RUN: %t.x86 || true
//
//---------------- merge the raw profiles ------------------------------------//
// RUN: llvm-profdata merge --object-aware-hashing=%t.x86 %t.x86.profraw \
// RUN:                     --object-aware-hashing=%t.hex %t.hex.profraw \
// RUN:                     -o %t.profdata
//
//---------------- show unified coverage & check -----------------------------//
// RUN: llvm-cov show -instr-profile=%t.profdata --object=%t.x86 --object=%t.hex --merge-binary-coverage | FileCheck %s
//
// CHECK-LABEL: {{.*fb\.c}}:
// CHECK:     1|      1|int foo() { return 0; }
// CHECK:     2|       |
// CHECK:     3|      1|int bar() { return 0; }
// CHECK:     4|       |
// CHECK:     5|      1|int bun() { return 0; }
//
// CHECK-LABEL: {{.*mypg\.c}}:
// CHECK:     1|      0|int baz() { return 0; }
// CHECK:     2|      1|int main() { return 1; }
//
// CHECK-LABEL: {{.*myprogram\.c}}:
// CHECK:     1|       |extern int foo();
// CHECK:     2|       |extern int bar();
// CHECK:     3|       |extern int bun();
// CHECK:     4|       |
// CHECK:     5|       |
// CHECK:     6|      1|int main()
// CHECK:     7|      1|  return foo() + bar() + bun();
// CHECK:     8|      1|}
