// Check that we combine expansion regions.

// RUN: llvm-profdata merge %S/Inputs/combine_expansions.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/combine_expansions.covmapping -instr-profile %t.profdata -filename-equivalence %s | FileCheck %s

#define SIMPLE_OP \
  ++x
// CHECK:       | [[@LINE-2]]|#define SIMPLE_OP
// CHECK-NEXT: 2| [[@LINE-2]]|  ++x

#define DO_SOMETHING \
  {                  \
    int x = 0;       \
    SIMPLE_OP;       \
  }
// CHECK:       | [[@LINE-5]]|#define DO_SOMETHING
// CHECK-NEXT: 2| [[@LINE-5]]|  {
// CHECK-NEXT: 2| [[@LINE-5]]|    int x = 0;
// CHECK-NEXT: 2| [[@LINE-5]]|    SIMPLE_OP;
// CHECK-NEXT: 2| [[@LINE-5]]|  }

int main() {    // CHECK:      1| [[@LINE]]|int main() {
  DO_SOMETHING; // CHECK-NEXT: 1| [[@LINE]]|  DO_SOMETHING;
  DO_SOMETHING; // CHECK-NEXT: 1| [[@LINE]]|  DO_SOMETHING;
  return 0;     // CHECK-NEXT: 1| [[@LINE]]|  return 0;
}               // CHECK-NEXT: 1| [[@LINE]]|}
