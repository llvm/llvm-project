// RUN: llvm-cov show %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata -dump -filename-equivalence %s 2>&1 | FileCheck %s

void func() {
  return;
  int i = 0;                     // CHECK: Highlighted line [[@LINE]], 3 -> ?
}                                // CHECK: Highlighted line [[@LINE]], 1 -> 2

void func2(int x) {
  if(x > 5) {
    while(x >= 9) {
      return;
      --x;                       // CHECK: Highlighted line [[@LINE]], 7 -> ?
    }                            // CHECK: Highlighted line [[@LINE]], 1 -> 6
    int i = 0;                   // CHECK: Highlighted line [[@LINE]], 5 -> ?
  }                              // CHECK: Highlighted line [[@LINE]], 1 -> 4
}

void test() {
  int x = 0;

  if (x) {                       // CHECK: Highlighted line [[@LINE]], 10 -> ?
    x = 0;                       // CHECK: Highlighted line [[@LINE]], 1 -> ?
  } else {                       // CHECK: Highlighted line [[@LINE]], 1 -> 4
    x = 1;
  }

                                  // CHECK: Highlighted line [[@LINE+1]], 26 -> 29
  for (int i = 0; i < 0; ++i) {   // CHECK: Highlighted line [[@LINE]], 31 -> ?
    x = 1;                        // CHECK: Highlighted line [[@LINE]], 1 -> ?
  }                               // CHECK: Highlighted line [[@LINE]], 1 -> 4

  x = x < 10 ? x +
               1
             : x - 1;             // CHECK: Highlighted line [[@LINE]], 16 -> 21
  x = x > 10 ? x +                // CHECK: Highlighted line [[@LINE]], 16 -> ?
               1                  // CHECK: Highlighted line [[@LINE]], 1 -> 17
             : x - 1;
}

int main() {
  test();
  func();
  func2(9);
  return 0;
}

// RUN: llvm-cov export %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata 2>&1 | FileCheck %S/Inputs/highlightedRanges.json
