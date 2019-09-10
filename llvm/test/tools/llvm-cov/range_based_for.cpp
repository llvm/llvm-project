// Make sure that compiler-added local variables (whose line number is zero)
// don't crash llvm-cov.




// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t
// RUN: cp %s %p/Inputs/range_based_for.gc* .

// RUN: llvm-cov gcov range_based_for.cpp | FileCheck %s --check-prefix=STDOUT
// STDOUT: File 'range_based_for.cpp'
// STDOUT: Lines executed:100.00% of 5
// STDOUT: range_based_for.cpp:creating 'range_based_for.cpp.gcov'

// RUN: FileCheck %s --check-prefix=GCOV < %t/range_based_for.cpp.gcov
// GCOV: -:    0:Runs:1
// GCOV: -:    0:Programs:1

int main(int argc, const char *argv[]) { // GCOV: 1:    [[@LINE]]:int main(
  int V[] = {1, 2};                      // GCOV: 1:    [[@LINE]]:  int V[]
  for (int &I : V) {                     // GCOV: 5:    [[@LINE]]:  for (
  }                                      // GCOV: 2:    [[@LINE]]:  }
  return 0;                              // GCOV: 1:    [[@LINE]]:  return
}                                        // GCOV: -:    [[@LINE]]:}

// llvm-cov doesn't work on big endian yet
// XFAIL: host-byteorder-big-endian
