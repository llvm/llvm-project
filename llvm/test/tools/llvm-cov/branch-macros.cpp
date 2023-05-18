// RUN: llvm-profdata merge %S/Inputs/branch-macros.proftext -o %t.profdata
// RUN: llvm-cov show --show-expansions --show-branches=count %S/Inputs/branch-macros.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov report --show-branch-summary %S/Inputs/branch-macros.o32l -instr-profile %t.profdata -show-functions -path-equivalence=/tmp,%S %s | FileCheck %s -check-prefix=REPORT

#define COND1 (a == b)
#define COND2 (a != b)
#define COND3 (COND1 && COND2)
#define COND4 (COND3 ? COND2 : COND1)
#define MACRO1 COND3
#define MACRO2 MACRO1
#define MACRO3 MACRO2

#include <stdlib.h>


bool func(int a, int b) {
  bool c = COND1 && COND2; // CHECK: |  |  |  Branch ([[@LINE-12]]:15): [True: 1, False: 2]
                           // CHECK: |  |  |  Branch ([[@LINE-12]]:15): [True: 0, False: 1]
  bool d = COND3;          // CHECK: |  |  |  |  |  Branch ([[@LINE-14]]:15): [True: 1, False: 2]
                           // CHECK: |  |  |  |  |  Branch ([[@LINE-14]]:15): [True: 0, False: 1]
  bool e = MACRO1;         // CHECK: |  |  |  |  |  |  |  Branch ([[@LINE-16]]:15): [True: 1, False: 2]
                           // CHECK: |  |  |  |  |  |  |  Branch ([[@LINE-16]]:15): [True: 0, False: 1]
  bool f = MACRO2;         // CHECK: |  |  |  |  |  |  |  |  |  Branch ([[@LINE-18]]:15): [True: 1, False: 2]
                           // CHECK: |  |  |  |  |  |  |  |  |  Branch ([[@LINE-18]]:15): [True: 0, False: 1]
  bool g = MACRO3;         // CHECK: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-20]]:15): [True: 1, False: 2]
                           // CHECK: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-20]]:15): [True: 0, False: 1]
  return c && d && e && f && g;
                           // CHECK: |  Branch ([[@LINE-1]]:10): [True: 0, False: 3]
                           // CHECK: |  Branch ([[@LINE-2]]:15): [True: 0, False: 0]
                           // CHECK: |  Branch ([[@LINE-3]]:20): [True: 0, False: 0]
                           // CHECK: |  Branch ([[@LINE-4]]:25): [True: 0, False: 0]
                           // CHECK: |  Branch ([[@LINE-5]]:30): [True: 0, False: 0]
}


bool func2(int a, int b) {
    bool h = MACRO3 || COND4;  // CHECK: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-32]]:15): [True: 1, False: 2]
                               // CHECK: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-32]]:15): [True: 0, False: 1]
                               // CHECK: |  |  |  |  |  |  |  Branch ([[@LINE-34]]:15): [True: 1, False: 2]
                               // CHECK: |  |  |  |  |  |  |  Branch ([[@LINE-34]]:15): [True: 0, False: 1]
                               // CHECK: |  |  |  Branch ([[@LINE-33]]:15): [True: 1, False: 2]
  return h;
}

extern "C" { extern void __llvm_profile_write_file(void); }
int main(int argc, char *argv[])
{
  func(atoi(argv[1]), atoi(argv[2]));
  func2(atoi(argv[1]), atoi(argv[2]));
  __llvm_profile_write_file();
  return 0;
}

// REPORT: Name                        Regions    Miss   Cover     Lines    Miss   Cover  Branches    Miss   Cover
// REPORT-NEXT: ---
// REPORT-NEXT: _Z4funcii                        28       4  85.71%        18       0 100.00%        30      14  53.33%
// REPORT-NEXT: _Z5func2ii                       13       1  92.31%         8       0 100.00%        10       2  80.00%
// REPORT-NEXT: main                              1       0 100.00%         6       0 100.00%         0       0   0.00%
// REPORT-NEXT: ---
// REPORT-NEXT: TOTAL                            42       5  88.10%        32       0 100.00% 40      16  60.00%
