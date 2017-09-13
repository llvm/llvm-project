// RUN: llvm-cov show -instr-profile %S/Inputs/ifdef.profdata %S/Inputs/ifdef.covmapping -dump -path-equivalence=/tmp,%S %s > %t.out 2>&1
// RUN: FileCheck %s -input-file %t.out -check-prefix=LINE
// RUN: FileCheck %s -input-file %t.out -check-prefix=HIGHLIGHT


int main() {
  if (0) { // LINE: [[@LINE]]|{{ +}}1|
#if 0      // LINE-NEXT: [[@LINE]]|{{ +}}|
#endif     // LINE-NEXT: [[@LINE]]|{{ +}}|
  }
  return 0;
}

// HIGHLIGHT: Highlighted line [[@LINE-7]], 10 -> ?
// HIGHLIGHT-NEXT: Highlighted line [[@LINE-7]], 1 -> 1
// HIGHLIGHT-NEXT: Highlighted line [[@LINE-6]], 1 -> 4
