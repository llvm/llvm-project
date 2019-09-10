// Make sure that compiler-added functions (whose line number is zero) don't
// crash llvm-cov.




// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t
// RUN: cp %s %p/Inputs/copy_block_helper.gc* .

// RUN: llvm-cov gcov copy_block_helper.m | FileCheck %s --check-prefix=STDOUT
// STDOUT: File 'copy_block_helper.m'
// STDOUT: Lines executed:100.00% of 5
// STDOUT: copy_block_helper.m:creating 'copy_block_helper.m.gcov'

// RUN: FileCheck %s --check-prefix=GCOV < %t/copy_block_helper.m.gcov
// GCOV: -:    0:Runs:1
// GCOV: -:    0:Programs:1

id test_helper(id (^foo)(void)) { return foo(); } // GCOV: 1:    [[@LINE]]:id
void test(id x) { // GCOV: -:    [[@LINE]]:void test
  test_helper(^{  // GCOV: 2:    [[@LINE]]:  test_helper
    return x;     // GCOV: 1:    [[@LINE]]:    return
  });             // GCOV: -:    [[@LINE]]:
}                 // GCOV: 1:    [[@LINE]]:}

// GCOV: 1:    [[@LINE+1]]:int main
int main(int argc, const char *argv[]) { test(0); }

// llvm-cov doesn't work on big endian yet
// XFAIL: host-byteorder-big-endian
