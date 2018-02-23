// RUN: llvm-cov report %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -filename-equivalence 2>&1 | FileCheck %s
// RUN: llvm-cov report -show-functions %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -filename-equivalence report.cpp 2>&1 | FileCheck -check-prefix=FILT %s
// RUN: llvm-cov report -show-functions %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -filename-equivalence report.cpp does-not-exist.cpp 2>&1 | FileCheck -check-prefix=FILT %s

// CHECK: Regions    Missed Regions     Cover   Functions  Missed Functions  Executed  Instantiations   Missed Insts.  Executed       Lines      Missed Lines     Cover
// CHECK-NEXT: ---
// CHECK-NEXT: report.cpp                          5                 2    60.00%           4                 1    75.00%               4               1    75.00%          13                 4    69.23%
// CHECK-NEXT: ---
// CHECK-NEXT: TOTAL                               5                 2    60.00%           4                 1    75.00%               4               1    75.00%          13                 4    69.23%

// FILT: File 'report.cpp':
// FILT-NEXT: Name        Regions  Miss   Cover  Lines  Miss   Cover
// FILT-NEXT: ---
// FILT-NEXT: _Z3foob           2     1  50.00%      4     2  50.00%
// FILT-NEXT: _Z3barv           1     0 100.00%      2     0 100.00%
// FILT-NEXT: _Z4funcv          1     1   0.00%      2     2   0.00%
// FILT-NEXT: main              1     0 100.00%      5     0 100.00%
// FILT-NEXT: ---
// FILT-NEXT: TOTAL             5     2  60.00%     13     4  69.23%

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
// RUN: llvm-cov show -o %t.dir %S/Inputs/report.covmapping -instr-profile=%S/Inputs/report.profdata -filename-equivalence -name-regex=".*" %s
// RUN: FileCheck -check-prefix=FUNCTIONS -input-file %t.dir/functions.txt %s
// FUNCTIONS: _Z3foob
// FUNCTIONS: _Z3barv
// FUNCTIONS: _Z4func
