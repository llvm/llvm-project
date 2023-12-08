// Check that we combine expansion regions.

// RUN: llvm-profdata merge %S/Inputs/combine_expansions.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/combine_expansions.covmapping -instr-profile %t.profdata -path-equivalence=/tmp/ec,%S %s | FileCheck %s

#define SIMPLE_OP \
  ++x
// CHECK:      [[@LINE-2]]|  |#define SIMPLE_OP
// CHECK-NEXT: [[@LINE-2]]| 2|  ++x

#define DO_SOMETHING \
  {                  \
    int x = 0;       \
    SIMPLE_OP;       \
  }
// CHECK:      [[@LINE-5]]|  |#define DO_SOMETHING
// CHECK-NEXT: [[@LINE-5]]| 2|  {
// CHECK-NEXT: [[@LINE-5]]| 2|    int x = 0;
// CHECK-NEXT: [[@LINE-5]]| 2|    SIMPLE_OP;
// CHECK-NEXT: [[@LINE-5]]| 2|  }

int main() {    // CHECK:      [[@LINE]]| 1|int main() {
  DO_SOMETHING; // CHECK-NEXT: [[@LINE]]| 1|  DO_SOMETHING;
  DO_SOMETHING; // CHECK-NEXT: [[@LINE]]| 1|  DO_SOMETHING;
  return 0;     // CHECK-NEXT: [[@LINE]]| 1|  return 0;
}               // CHECK-NEXT: [[@LINE]]| 1|}
