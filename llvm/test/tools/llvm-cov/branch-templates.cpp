// RUN: llvm-profdata merge %S/Inputs/branch-templates.proftext -o %t.profdata
// RUN: llvm-cov show --show-expansions --show-branches=count %S/Inputs/branch-templates.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov report --show-branch-summary %S/Inputs/branch-templates.o32l -instr-profile %t.profdata -show-functions -path-equivalence=/tmp,%S %s | FileCheck %s -check-prefix=REPORT
// RUN: llvm-cov report --show-branch-summary %S/Inputs/branch-templates.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s -check-prefix=REPORTFILE

#include <stdio.h>
template<typename T>
void unused(T x) {
  return;
}

template<typename T>
int func(T x) {
  if(x)       // CHECK: |  Branch ([[@LINE]]:6): [True: 0, False: 1]
    return 0; // CHECK: |  Branch ([[@LINE-1]]:6): [True: 1, False: 0]
  else        // CHECK: |  Branch ([[@LINE-2]]:6): [True: 0, False: 1]
    return 1;
  int j = 1;
}

              // CHECK-LABEL: _Z4funcIiEiT_:
              // CHECK: |  |  Branch ([[@LINE-8]]:6): [True: 0, False: 1]
              // CHECK-LABEL: _Z4funcIbEiT_:
              // CHECK: |  |  Branch ([[@LINE-10]]:6): [True: 1, False: 0]
              // CHECK-LABEL: _Z4funcIfEiT_:
              // CHECK: |  |  Branch ([[@LINE-12]]:6): [True: 0, False: 1]

extern "C" { extern void __llvm_profile_write_file(void); }
int main() {
  if (func<int>(0))      // CHECK: |  Branch ([[@LINE]]:7): [True: 1, False: 0]
    printf("case1\n");
  if (func<bool>(true))  // CHECK: |  Branch ([[@LINE]]:7): [True: 0, False: 1]
    printf("case2\n");
  if (func<float>(0.0))  // CHECK: |  Branch ([[@LINE]]:7): [True: 1, False: 0]
    printf("case3\n");
  __llvm_profile_write_file();
  return 0;
}

// REPORT: Name                        Regions    Miss   Cover     Lines    Miss   Cover  Branches    Miss   Cover
// REPORT-NEXT: ---
// REPORT-NEXT: main                              7       1  85.71%        10       1  90.00%         6       3  50.00%
// REPORT-NEXT: _Z4funcIiEiT_                     5       2  60.00%         7       3  57.14%         2       1  50.00%
// REPORT-NEXT: _Z4funcIbEiT_                     5       2  60.00%         7       4  42.86%         2       1  50.00%
// REPORT-NEXT: _Z4funcIfEiT_                     5       2  60.00%         7       3  57.14%         2       1  50.00%
// REPORT-NEXT: ---
// REPORT-NEXT: TOTAL                            22       7  68.18%        31      11  64.52%        12       6  50.00%

// Make sure the covered branch tally for the function instantiation group is
// merged to reflect maximum branch coverage of a single instantiation, just
// like what is done for lines and regions. Also, the total branch tally
// summary for an instantiation group should agree with the total number of
// branches in the definition (In this case, 2 and 6 for func<>() and main(),
// respectively).  This is returned by: FunctionCoverageSummary::get(const
// InstantiationGroup &Group, ...)

// REPORTFILE: Filename                      Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       Lines      Missed Lines     Cover    Branches   Missed Branches     Cover
// REPORTFILE-NEXT: ---
// REPORTFILE-NEXT: branch-templates.cpp          12                 3    75.00%           2                 0   100.00%          17                 4    76.47%           8                 4    50.00%
// REPORTFILE-NEXT: ---
// REPORTFILE-NEXT: TOTAL                              12                 3    75.00%           2                 0   100.00%          17                 4    76.47%           8                 4    50.00%
