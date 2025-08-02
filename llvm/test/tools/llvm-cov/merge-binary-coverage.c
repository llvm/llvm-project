// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping -DTOGGLE %s \
// RUN:       -o %t.toggle
// RUN: env LLVM_PROFILE_FILE=%t.toggle.profraw 
// RUN: %t.toggle
// RUN: cp default.profraw %t.toggle.profraw

//------------------ build & run variant WITHOUT TOGGLE ----------------------//
// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping %s \
// RUN:       -o %t.notoggle
// RUN: env LLVM_PROFILE_FILE=%t.notoggle.profraw 
// RUN: %t.notoggle

//------------------ merge profiles & show coverage --------------------------//
// RUN: llvm-profdata merge --object-aware-hashing=%t.notoggle %t.notoggle.profraw\
// RUN:     --object-aware-hashing=%t.toggle   %t.toggle.profraw \
// RUN:     -o %t.profdata
//
// RUN: llvm-cov show -instr-profile=%t.profdata                     \
// RUN:       --object=%t.toggle --object=%t.notoggle                \
// RUN:       --merge-binary-coverage                                \
// RUN:   | FileCheck %s

/* Expected unified coverage table:

 1|      2|int main() {
 2|      2|  int a = 1;
 3|      2|  int b = 2;
 4|      2|  int res = 0;
 5|      2|#if defined(TOGGLE)
 6|      1|    res = a + b;
 7|      1|#else
 8|      1|    res = b - a;
 9|      1|#endif
10|      2|  return 0;
11|      2|}
*/

// CHECK-LABEL: {{^ *1\|}}
// CHECK: 1|      2|int main()
// CHECK: 2|      2|int a = 1;
// CHECK: 3|      2|int b = 2;
// CHECK: 4|      2|int res = 0;
// CHECK: 5|      2|#if defined(TOGGLE)
// CHECK: 6|      1|    res = a + b;
// CHECK: 7|      1|#else
// CHECK: 8|      1|    res = b - a;
// CHECK: 9|      1|#endif
// CHECK: 10|      2|    return 0;
// CHECK: 11|      2|}

int main() {
  int a = 1;
  int b = 2;
  int res = 0;
#if defined(TOGGLE)
  res = a + b;
#else
  res = b - a;
#endif
  return 0;
}