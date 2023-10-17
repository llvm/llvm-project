// RUN: llvm-cov report %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -path-equivalence=/tmp,%S 2>&1 -show-region-summary -show-instantiation-summary | FileCheck %s
// RUN: llvm-cov report -show-functions %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -path-equivalence=/tmp,%S %s 2>&1 | FileCheck -check-prefix=FILT %s
// RUN: llvm-cov report -show-functions %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -path-equivalence=/tmp,%S %s does-not-exist.cpp 2>&1 | FileCheck -check-prefix=FILT %s
// RUN: not llvm-cov report -show-functions %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -path-equivalence=/tmp,%S 2>&1 | FileCheck -check-prefix=NO_FILES %s

// NO_FILES: source files must be specified when -show-functions=true is specified

// CHECK: Regions    Missed Regions     Cover   Functions  Missed Functions  Executed  Instantiations   Missed Insts.  Executed       Lines      Missed Lines     Cover
// CHECK-NEXT: ---
// CHECK-NEXT: report.cpp                          6                 2    66.67%           4                 1    75.00%               4               1    75.00%          13                 3    76.92%
// CHECK-NEXT: ---
// CHECK-NEXT: TOTAL                               6                 2    66.67%           4                 1    75.00%               4               1    75.00%          13                 3    76.92%

// FILT: File '{{.*}}report.cpp':
// FILT-NEXT: Name        Regions  Miss   Cover  Lines  Miss   Cover
// FILT-NEXT: ---
// FILT-NEXT: _Z3foob           3     1  66.67%      4     1  75.00%
// FILT-NEXT: _Z3barv           1     0 100.00%      2     0 100.00%
// FILT-NEXT: _Z4funcv          1     1   0.00%      2     2   0.00%
// FILT-NEXT: main              1     0 100.00%      5     0 100.00%
// FILT-NEXT: ---
// FILT-NEXT: TOTAL             6     2  66.67%     13     3  76.92%

void foo(bool cond) {
  if (cond) {
  }
}

void bar() {
}

void func() {
}

int main() {
  foo(false);
  bar();
  return 0;
}

// Test that listing multiple functions in a function view works.
// RUN: llvm-cov show -o %t.dir %S/Inputs/report.covmapping -instr-profile=%S/Inputs/report.profdata -path-equivalence=/tmp,%S -name-regex=".*" %s
// RUN: FileCheck -check-prefix=FUNCTIONS -input-file %t.dir/coverage/tmp/report.cpp.txt %s
// FUNCTIONS: _Z3foob
// FUNCTIONS: _Z3barv
// FUNCTIONS: _Z4func
