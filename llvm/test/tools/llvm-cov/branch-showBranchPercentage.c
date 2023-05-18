// Test visualization of branch taken percentages

// RUN: llvm-profdata merge %S/Inputs/branch-showBranchPercentage.proftext -o %t.profdata
// RUN: llvm-cov show --show-branches=percent %S/Inputs/branch-showBranchPercentage.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

extern void __llvm_profile_write_file(void);

int main(int argc, char *argv[])
{
  int i = 0;
  if (argc < 3)                       // CHECK: Branch ([[@LINE]]:7): [True: 16.67%, False: 83.33%]
  {
    __llvm_profile_write_file();
    return 0;
  }

  int a = atoi(argv[1]);
  int b = atoi(argv[2]);

                                      // CHECK: Branch ([[@LINE+4]]:8): [True: 20.00%, False: 80.00%]
                                      // CHECK: Branch ([[@LINE+3]]:18): [True: 0.00%, False: 100.00%]
                                      // CHECK: Branch ([[@LINE+2]]:29): [True: 0.00%, False: 100.00%]
                                      // CHECK: Branch ([[@LINE+1]]:40): [True: 40.00%, False: 60.00%]
  if ((a == 0 && b == 2) || b == 34 || a == b)
    printf("case1\n");

  b = (a != 0 || a == 2) ? b : b+2;   // CHECK: Branch ([[@LINE]]:8): [True: 80.00%, False: 20.00%]
                                      // CHECK: Branch ([[@LINE-1]]:18): [True: 0.00%, False: 100.00%]
  b = (a != 0 && a == 1);             // CHECK: Branch ([[@LINE]]:8): [True: 80.00%, False: 20.00%]
                                      // CHECK: Branch ([[@LINE-1]]:18): [True: 25.00%, False: 75.00%]
  for (i = 0; i < b; i++) { a = 2 + b + b; }
                                      // CHECK: Branch ([[@LINE-1]]:15): [True: 16.67%, False: 83.33%]

  b = a;

  switch (a)
  {
    case 0:                           // CHECK: Branch ([[@LINE]]:5): [True: 20.00%, False: 80.00%]
      printf("case0\n");
    case 2:                           // CHECK: Branch ([[@LINE]]:5): [True: 20.00%, False: 80.00%]
      printf("case2\n");
    case 3:                           // CHECK: Branch ([[@LINE]]:5): [True: 0.00%, False: 100.00%]
      printf("case3\n");
    default: break;                   // CHECK: Branch ([[@LINE]]:5): [True: 60.00%, False: 40.00%]
  }

  i = 0;
  do {
    printf("loop\n");
  } while (i++ < 10);                 // CHECK: Branch ([[@LINE]]:12): [True: 90.91%, False: 9.09%]

  __llvm_profile_write_file();

  return b;
}
// RUN: llvm-profdata merge %S/Inputs/branch-showBranchPercentage.proftext -o %t.profdata
// RUN: llvm-cov show --show-branches=percent %S/Inputs/branch-showBranchPercentage.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s -format html -o %t.html.dir

// Test html output.
// RUN: FileCheck -check-prefix=HTML -input-file=%t.html.dir/coverage/tmp/branch-showBranchPercentage.c.html %s
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}16.67%,{{.*}}83.33%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}20.00%,{{.*}}80.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}0.00%,{{.*}}100.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}0.00%,{{.*}}100.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}40.00%,{{.*}}60.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}80.00%,{{.*}}20.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}0.00%,{{.*}}100.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}80.00%,{{.*}}20.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}25.00%,{{.*}}75.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}16.67%,{{.*}}83.33%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}20.00%,{{.*}}80.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}20.00%,{{.*}}80.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}0.00%,{{.*}}100.00%]
// HTML: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>{{.*}}60.00%,{{.*}}40.00%]
