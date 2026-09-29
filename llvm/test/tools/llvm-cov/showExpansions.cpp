// RUN: llvm-cov show %S/Inputs/showExpansions.covmapping -instr-profile %S/Inputs/showExpansions.profdata -dump -show-expansions -path-equivalence="/Users/bogner/code/llvm/test/tools,%S/.." %s 2>&1 | FileCheck %s

#define DO_SOMETHING_ELSE() \
  do {                      \
  } while (0)
#define ANOTHER_THING() \
  do {                  \
    if (0) {            \
    }                   \
  } while (0)

#define DO_SOMETHING(x)    \
  do {                     \
    if (x)                 \
      DO_SOMETHING_ELSE(); \
    else                   \
      ANOTHER_THING();     \
  } while (0)
// CHECK-DAG: Expansion at line [[@LINE-4]], 7 -> 24
// CHECK-DAG: Expansion at line [[@LINE-3]], 7 -> 20

int main(int argc, const char *argv[]) {
  for (int i = 0; i < 100; ++i)
    DO_SOMETHING(i); // CHECK-DAG: Expansion at line [[@LINE]], 5 -> 17
  return 0;
}
// RUN: llvm-cov export %S/Inputs/showExpansions.covmapping -instr-profile %S/Inputs/showExpansions.profdata 2>&1 | FileCheck %S/Inputs/showExpansions.json

// RUN: llvm-cov export %S/Inputs/showExpansions.covmapping -instr-profile %S/Inputs/showExpansions.profdata -skip-expansions 2>&1 | FileCheck %s -check-prefix=SKIP-EXPANSIONS
// SKIP-EXPANSIONS-NOT: "expansions"
