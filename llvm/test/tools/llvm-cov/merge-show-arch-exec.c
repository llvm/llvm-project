// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping -DTOGGLE %s \
// RUN:       -o %t.toggle
// RUN: %t.toggle
// RUN: cp default.profraw %t.toggle.profraw

//------------------ build & run variant WITHOUT TOGGLE ----------------------//
// RUN: clang -O0 -fprofile-instr-generate -fcoverage-mapping %s \
// RUN:       -o %t.notoggle
// RUN: %t.notoggle
// RUN: cp default.profraw %t.notoggle.profraw 

//------------------ merge profiles & show coverage --------------------------//
// RUN: llvm-profdata merge --object-aware-hashing=%t.notoggle %t.notoggle.profraw\
// RUN:     --object-aware-hashing=%t.toggle   %t.toggle.profraw \
// RUN:     -o %t.profdata
//
// RUN: llvm-cov show -instr-profile=%t.profdata --object=%t.toggle --object=%t.notoggle --merge-binary-coverage --show-arch-executables | FileCheck %s

// CHECK:       |    1|      2|int main() {
// CHECK:       |    2|      2|int a = 1;
// CHECK:       |    3|      2|int b = 2;
// CHECK:       |    4|      2|int res = 0;
// CHECK:       |    5|      2|#if defined\(TOGGLE\)
// CHECK:       |    6|      1|    res = a \+ b;
// CHECK:       |    7|      1|#else
// CHECK:       |    8|      1|    res = b - a;
// CHECK:       |    9|      1|#endif
// CHECK:       |   10|      2|    return 0;
// CHECK:       |   11|      2|}
// CHECK:       |  ------------------
// CHECK:       |  | main:
// CHECK:       |  |    -x86_64
// CHECK:       |  |    -merge-ifdef.toggle:
// CHECK:       |  |    1|      1|int main\(\) {
// CHECK:       |  |    2|      1|int a = 1;
// CHECK:       |  |    3|      1|int b = 2;
// CHECK:       |  |    4|      1|int res = 0;
// CHECK:       |  |    5|      1|#if defined\(TOGGLE\)
// CHECK:       |  |    6|      1|    res = a \+ b;
// CHECK:       |  |    7|       |#else
// CHECK:       |  |    8|       |    res = b - a;
// CHECK:       |  |    9|       |#endif
// CHECK:       |  |   10|      1|    return 0;
// CHECK:       |  |   11|      1|}
// CHECK:       |  ------------------
// CHECK:       |  | main:
// CHECK:       |  |    -x86_64
// CHECK:       |  |    -merge-ifdef.notoggle:
// CHECK:       |  |    1|      1|int main\(\) {
// CHECK:       |  |    2|      1|int a = 1;
// CHECK:       |  |    3|      1|int b = 2;
// CHECK:       |  |    4|      1|int res = 0;
// CHECK:       |  |    5|       |#if defined\(TOGGLE\)
// CHECK:       |  |    6|       |    res = a \+ b;
// CHECK:       |  |    7|       |#else
// CHECK:       |  |    8|      1|    res = b - a;
// CHECK:       |  |    9|      1|#endif
// CHECK:       |  |   10|      1|    return 0;
// CHECK:       |  |   11|      1|}
// CHECK:       |  ------------------



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