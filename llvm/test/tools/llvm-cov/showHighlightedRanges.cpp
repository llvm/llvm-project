// RUN: llvm-cov show %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata -dump -path-equivalence=/Users/bogner/code/llvm/test/tools,%S/.. %s 2>&1 | FileCheck %s -check-prefixes=TEXT,SHARED

void func() {
  return;
  int i = 0;                     // SHARED: Highlighted line [[@LINE]], 3 ->
}                                // SHARED: Highlighted line [[@LINE]], 1 -> 2

void func2(int x) {
  if(x > 5) {
    while(x >= 9) {
      return;
      --x;                       // SHARED: Highlighted line [[@LINE]], 7 ->
    }
    int i = 0;                   // SHARED: Highlighted line [[@LINE]], 5 ->
  }                              // SHARED: Highlighted line [[@LINE]], 1 -> 4
}

void test() {
  int x = 0;

  if (x) {                       // SHARED: Highlighted line [[@LINE]], 10 ->
    x = 0;                       // SHARED: Highlighted line [[@LINE]], 1 ->
  } else {                       // SHARED: Highlighted line [[@LINE]], 1 -> 4
    x = 1;
  }

                                  // SHARED: Highlighted line [[@LINE+1]], 26 ->
  for (int i = 0; i < 0; ++i) {   // SHARED: Highlighted line [[@LINE]], 31 ->
    x = 1;                        // SHARED: Highlighted line [[@LINE]], 1 ->
  }                               // SHARED: Highlighted line [[@LINE]], 1 -> 4

  x = x < 10 ? x +
               1
             : x - 1;             // SHARED: Highlighted line [[@LINE]], 16 -> 21
  x = x > 10 ? x +                // SHARED: Highlighted line [[@LINE]], 16 ->
               1                  // SHARED: Highlighted line [[@LINE]], 1 -> 17
             : x - 1;
}

int main() {
  test();
  func();
  func2(9);
  return 0;
}

// RUN: llvm-cov show %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata -format html -dump -path-equivalence=/Users/bogner/code/llvm/test/tools,%S/.. %s 2>&1 | FileCheck %s -check-prefixes=HTML,SHARED
// RUN: llvm-cov export %S/Inputs/highlightedRanges.covmapping -instr-profile %S/Inputs/highlightedRanges.profdata 2>&1 | FileCheck %S/Inputs/highlightedRanges.json
