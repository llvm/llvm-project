// RUN: llvm-cov show %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata -dump -filename-equivalence %s 2>&1 | FileCheck %s -check-prefixes=TEXT,SHARED

void func() {
  return;                        // TEXT: Highlighted line [[@LINE+1]], 3 -> ?
  int i = 0;                     // HTML: Highlighted line [[@LINE]], 1 ->
}                                // SHARED: Highlighted line [[@LINE]], 1 -> 2

void func2(int x) {
  if(x > 5) {
    while(x >= 9) {
      return;
      --x;                       // TEXT: Highlighted line [[@LINE]], 7 -> ?
    }                            // SHARED: Highlighted line [[@LINE]], 1 -> 6
    int i = 0;                   // TEXT: Highlighted line [[@LINE]], 5 -> ?
  }                              // SHARED: Highlighted line [[@LINE]], 1 -> 4
}

void test() {
  int x = 0;

  if (x) {                       // TEXT: Highlighted line [[@LINE]], 10 -> ?
    x = 0;                       // SHARED: Highlighted line [[@LINE]], 1 -> ?
  } else {                       // TEXT: Highlighted line [[@LINE]], 1 -> 4
    x = 1;
  }

                                  // TEXT: Highlighted line [[@LINE+1]], 26 -> 29
  for (int i = 0; i < 0; ++i) {   // TEXT: Highlighted line [[@LINE]], 31 -> ?
    x = 1;                        // TEXT: Highlighted line [[@LINE]], 1 -> ?
  }                               // SHARED: Highlighted line [[@LINE]], 1 -> 4

  x = x < 10 ? x +
               1
             : x - 1;             // TEXT: Highlighted line [[@LINE]], 16 -> 21
  x = x > 10 ? x +                // TEXT: Highlighted line [[@LINE]], 16 -> ?
               1                  // SHARED: Highlighted line [[@LINE]], 1 -> 17
             : x - 1;
}

int main() {
  test();
  func();
  func2(9);
  return 0;
}

// RUN: llvm-cov show %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata -format html -dump -filename-equivalence %s 2>&1 | FileCheck %s -check-prefixes=HTML,SHARED
// RUN: llvm-cov export %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata 2>&1 | FileCheck %S/Inputs/highlightedRanges.json
