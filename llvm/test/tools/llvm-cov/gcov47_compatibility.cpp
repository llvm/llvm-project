// Make sure that llvm-cov can read coverage data written in gcov47+ compatible
// format.

// Compile with these arguments and run the result to generate .gc* files:
// -coverage -Xclang -coverage-no-function-names-in-data
// -Xclang -coverage-cfg-checksum -Xclang -coverage-version='407*'




// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t
// RUN: cp %s %p/Inputs/gcov47_compatibility.gc* .

// RUN: llvm-cov gcov gcov47_compatibility.cpp | FileCheck %s --check-prefix=STDOUT
// STDOUT: File 'gcov47_compatibility.cpp'
// STDOUT: Lines executed:100.00% of 1
// STDOUT: gcov47_compatibility.cpp:creating 'gcov47_compatibility.cpp.gcov'

// RUN: FileCheck %s --check-prefix=GCOV < %t/gcov47_compatibility.cpp.gcov
// GCOV: -:    0:Runs:1
// GCOV: -:    0:Programs:1

int main(int argc, const char *argv[]) { // GCOV: -:    [[@LINE]]:int main(
  return 0;                              // GCOV: 1:    [[@LINE]]:  return
}                                        // GCOV: -:    [[@LINE]]:}

// llvm-cov doesn't work on big endian yet
// XFAIL: host-byteorder-big-endian
