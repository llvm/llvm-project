// Test visualization of general branch constructs in C.

// RUN: llvm-profdata merge %S/Inputs/branch-c-general.proftext -o %t.profdata
// RUN: llvm-cov show --show-branches=count %S/Inputs/branch-c-general.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov report --show-branch-summary %S/Inputs/branch-c-general.o32l -instr-profile %t.profdata -show-functions -path-equivalence=/tmp,%S %s | FileCheck %s -check-prefix=REPORT

void simple_loops() {
  int i;
  for (i = 0; i < 100; ++i) {    // CHECK: Branch ([[@LINE]]:15): [True: 100, False: 1]
  }
  while (i > 0)                  // CHECK: Branch ([[@LINE]]:10): [True: 100, False: 1]
    i--;
  do {} while (i++ < 75);        // CHECK: Branch ([[@LINE]]:16): [True: 75, False: 1]

}

void conditionals() {
  for (int i = 0; i < 100; ++i) {// CHECK: Branch ([[@LINE]]:19): [True: 100, False: 1]
    if (i % 2) {                 // CHECK: Branch ([[@LINE]]:9): [True: 50, False: 50]
      if (i) {}                  // CHECK: Branch ([[@LINE]]:11): [True: 50, False: 0]
    } else if (i % 3) {          // CHECK: Branch ([[@LINE]]:16): [True: 33, False: 17]
      if (i) {}                  // CHECK: Branch ([[@LINE]]:11): [True: 33, False: 0]
    } else {
      if (i) {}                  // CHECK: Branch ([[@LINE]]:11): [True: 16, False: 1]
    }
                                 // CHECK: Branch ([[@LINE+1]]:9): [Folded - Ignored]
    if (1 && i) {}               // CHECK: Branch ([[@LINE]]:14): [True: 99, False: 1]
    if (0 || i) {}               // CHECK: Branch ([[@LINE]]:9): [Folded - Ignored]
  }                              // CHECK: Branch ([[@LINE-1]]:14): [True: 99, False: 1]

}

void early_exits() {
  int i = 0;

  if (i) {}                     // CHECK: Branch ([[@LINE]]:7): [True: 0, False: 1]

  while (i < 100) {             // CHECK: Branch ([[@LINE]]:10): [True: 51, False: 0]
    i++;
    if (i > 50)                 // CHECK: Branch ([[@LINE]]:9): [True: 1, False: 50]
      break;
    if (i % 2)                  // CHECK: Branch ([[@LINE]]:9): [True: 25, False: 25]
      continue;
  }

  if (i) {}                     // CHECK: Branch ([[@LINE]]:7): [True: 1, False: 0]

  do {
    if (i > 75)                 // CHECK: Branch ([[@LINE]]:9): [True: 1, False: 25]
      return;
    else
      i++;
  } while (i < 100);            // CHECK: Branch ([[@LINE]]:12): [True: 25, False: 0]

  if (i) {}                     // CHECK: Branch ([[@LINE]]:7): [True: 0, False: 0]

}

void jumps() {
  int i;

  for (i = 0; i < 2; ++i) {     // CHECK: Branch ([[@LINE]]:15): [True: 1, False: 0]
    goto outofloop;
    // Never reached -> no weights
    if (i) {}                   // CHECK: Branch ([[@LINE]]:9): [True: 0, False: 0]
  }

outofloop:
  if (i) {}                     // CHECK: Branch ([[@LINE]]:7): [True: 0, False: 1]

  goto loop1;

  while (i) {                   // CHECK: Branch ([[@LINE]]:10): [True: 0, False: 1]
  loop1:
    if (i) {}                   // CHECK: Branch ([[@LINE]]:9): [True: 0, False: 1]
  }

  goto loop2;
first:
second:
third:
  i++;
  if (i < 3)                    // CHECK: Branch ([[@LINE]]:7): [True: 2, False: 1]
    goto loop2;

  while (i < 3) {               // CHECK: Branch ([[@LINE]]:10): [True: 0, False: 1]
  loop2:
    switch (i) {
    case 0:                     // CHECK: Branch ([[@LINE]]:5): [True: 1, False: 2]
      goto first;
    case 1:                     // CHECK: Branch ([[@LINE]]:5): [True: 1, False: 2]
      goto second;
    case 2:                     // CHECK: Branch ([[@LINE]]:5): [True: 1, False: 2]
      goto third;
    }
  }

  for (i = 0; i < 10; ++i) {    // CHECK: Branch ([[@LINE]]:15): [True: 10, False: 1]
    goto withinloop;
    // never reached -> no weights
    if (i) {}                   // CHECK: Branch ([[@LINE]]:9): [True: 0, False: 0]
  withinloop:
    if (i) {}                   // CHECK: Branch ([[@LINE]]:9): [True: 9, False: 1]
  }

}

void switches() {
  static int weights[] = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5};

  // No cases -> no weights
  switch (weights[0]) {
  default:                      // CHECK: Branch ([[@LINE]]:3): [True: 1, False: 0]
    break;
  }
                                // CHECK: Branch ([[@LINE+1]]:63): [True: 15, False: 0]
  for (int i = 0, len = sizeof(weights) / sizeof(weights[0]); i < len; ++i) {
    switch (i[weights]) {
    case 1:                     // CHECK: Branch ([[@LINE]]:5): [True: 1, False: 14]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 0, False: 1]
      // fallthrough
    case 2:                     // CHECK: Branch ([[@LINE]]:5): [True: 2, False: 13]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 2, False: 1]
      break;
    case 3:                     // CHECK: Branch ([[@LINE]]:5): [True: 3, False: 12]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 3, False: 0]
      continue;
    case 4:                     // CHECK: Branch ([[@LINE]]:5): [True: 4, False: 11]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 4, False: 0]
      switch (i) {
      case 6 ... 9:             // CHECK: Branch ([[@LINE]]:7): [True: 4, False: 0]
        if (i) {}               // CHECK: Branch ([[@LINE]]:13): [True: 4, False: 0]
        continue;
      }

    default:                    // CHECK: Branch ([[@LINE]]:5): [True: 5, False: 10]
      if (i == len - 1)         // CHECK: Branch ([[@LINE]]:11): [True: 1, False: 4]
        return;
    }
  }

  // Never reached -> no weights
  if (weights[0]) {}            // CHECK: Branch ([[@LINE]]:7): [True: 0, False: 0]

}

void big_switch() {
  for (int i = 0; i < 32; ++i) {// CHECK: Branch ([[@LINE]]:19): [True: 32, False: 1]
    switch (1 << i) {
    case (1 << 0):              // CHECK: Branch ([[@LINE]]:5): [True: 1, False: 31]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 0, False: 1]
      // fallthrough
    case (1 << 1):              // CHECK: Branch ([[@LINE]]:5): [True: 1, False: 31]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 1, False: 1]
      break;
    case (1 << 2) ... (1 << 12):// CHECK: Branch ([[@LINE]]:5): [True: 11, False: 21]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 11, False: 0]
      break;
      // The branch for the large case range above appears after the case body.

    case (1 << 13):             // CHECK: Branch ([[@LINE]]:5): [True: 1, False: 31]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 1, False: 0]
      break;
    case (1 << 14) ... (1 << 28):// CHECK: Branch ([[@LINE]]:5): [True: 15, False: 17]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 15, False: 0]
      break;
    // The branch for the large case range above appears after the case body.
    // CHECK: Branch ([[@LINE+1]]:5): [True: 1, False: 31]
    case (1 << 29) ... ((1 << 29) + 1):
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 1, False: 0]
      break;
    default:                    // CHECK: Branch ([[@LINE]]:5): [True: 2, False: 30]
      if (i) {}                 // CHECK: Branch ([[@LINE]]:11): [True: 2, False: 0]
      break;
    }
  }

}

void boolean_operators() {
  int v;                        // CHECK: Branch ([[@LINE+1]]:19): [True: 100, False: 1]
  for (int i = 0; i < 100; ++i) {
    v = i % 3 || i;             // CHECK: Branch ([[@LINE]]:9): [True: 66, False: 34]
                                // CHECK: Branch ([[@LINE-1]]:18): [True: 33, False: 1]
    v = i % 3 && i;             // CHECK: Branch ([[@LINE]]:9): [True: 66, False: 34]
                                // CHECK: Branch ([[@LINE-1]]:18): [True: 66, False: 0]
    v = i % 3 || i % 2 || i;    // CHECK: Branch ([[@LINE]]:9): [True: 66, False: 34]
                                // CHECK: Branch ([[@LINE-1]]:18): [True: 17, False: 17]
    v = i % 2 && i % 3 && i;    // CHECK: Branch ([[@LINE-2]]:27): [True: 16, False: 1]
  }                             // CHECK: Branch ([[@LINE-1]]:9): [True: 50, False: 50]
                                // CHECK: Branch ([[@LINE-2]]:18): [True: 33, False: 17]
}                               // CHECK: Branch ([[@LINE-3]]:27): [True: 33, False: 0]

void boolop_loops() {
  int i = 100;

  while (i && i > 50)           // CHECK: Branch ([[@LINE]]:10): [True: 51, False: 0]
    i--;                        // CHECK: Branch ([[@LINE-1]]:15): [True: 50, False: 1]

  while ((i % 2) || (i > 0))    // CHECK: Branch ([[@LINE]]:10): [True: 25, False: 26]
    i--;                        // CHECK: Branch ([[@LINE-1]]:21): [True: 25, False: 1]

  for (i = 100; i && i > 50; --i); // CHECK: Branch ([[@LINE]]:17): [True: 51, False: 0]
                                   // CHECK: Branch ([[@LINE-1]]:22): [True: 50, False: 1]
  for (; (i % 2) || (i > 0); --i); // CHECK: Branch ([[@LINE]]:10): [True: 25, False: 26]
                                   // CHECK: Branch ([[@LINE-1]]:21): [True: 25, False: 1]
}

void conditional_operator() {
  int i = 100;

  int j = i < 50 ? i : 1;       // CHECK: Branch ([[@LINE]]:11): [True: 0, False: 1]

  int k = i ?: 0;               // CHECK: Branch ([[@LINE]]:11): [True: 1, False: 0]

}

void do_fallthrough() {
  for (int i = 0; i < 10; ++i) {// CHECK: Branch ([[@LINE]]:19): [True: 10, False: 1]
    int j = 0;
    do {
      // The number of exits out of this do-loop via the break statement
      // exceeds the counter value for the loop (which does not include the
      // fallthrough count). Make sure that does not violate any assertions.
      if (i < 8) break;         // CHECK: Branch ([[@LINE]]:11): [True: 8, False: 4]
      j++;
    } while (j < 2);            // CHECK: Branch ([[@LINE]]:14): [True: 2, False: 2]
  }
}

static void static_func() {
  for (int i = 0; i < 10; ++i) {// CHECK: Branch ([[@LINE]]:19): [True: 10, False: 1]
  }
}










int main(int argc, const char *argv[]) {
  simple_loops();
  conditionals();
  early_exits();
  jumps();
  switches();
  big_switch();
  boolean_operators();
  boolop_loops();
  conditional_operator();
  do_fallthrough();
  static_func();
  extern void __llvm_profile_write_file();
  __llvm_profile_write_file();
  return 0;
}

// REPORT: Name                        Regions    Miss   Cover     Lines    Miss   Cover  Branches    Miss   Cover
// REPORT-NEXT: ---
// REPORT-NEXT: simple_loops                      8       0 100.00%         9       0 100.00%         6       0 100.00%
// REPORT-NEXT: conditionals                     24       0 100.00%        15       0 100.00%        16       2  87.50%
// REPORT-NEXT: early_exits                      20       4  80.00%        25       3  88.00%        16       6  62.50%
// REPORT-NEXT: jumps                            39      12  69.23%        48       4  91.67%        26       9  65.38%
// REPORT-NEXT: switches                         28       5  82.14%        38       5  86.84%        30       9  70.00%
// REPORT-NEXT: big_switch                       25       1  96.00%        32       0 100.00%        30       6  80.00%
// REPORT-NEXT: boolean_operators                16       0 100.00%        13       0 100.00%        22       2  90.91%
// REPORT-NEXT: boolop_loops                     19       0 100.00%        14       0 100.00%        16       2  87.50%
// REPORT-NEXT: conditional_operator              4       2  50.00%         8       1  87.50%         4       2  50.00%
// REPORT-NEXT: do_fallthrough                    9       0 100.00%        12       0 100.00%         6       0 100.00%
// REPORT-NEXT: main                              1       0 100.00%        16       0 100.00%         0       0   0.00%
// REPORT-NEXT: c-general.c:static_func           4       0 100.00%         4       0 100.00%         2       0 100.00%
// REPORT-NEXT: ---
// REPORT-NEXT: TOTAL                           197      24  87.82%       234      13 94.44%       174      38  78.16%

// Test file-level report.
// RUN: llvm-profdata merge %S/Inputs/branch-c-general.proftext -o %t.profdata
// RUN: llvm-cov report %S/Inputs/branch-c-general.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s -check-prefix=FILEREPORT
// FILEREPORT: TOTAL{{.*}}174                38    78.16%

// Test color True/False output.
// RUN: llvm-cov show --use-color --show-branches=count %S/Inputs/branch-c-general.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s -check-prefix=USECOLOR
// USECOLOR: Branch ({{[0-9]+}}:7): {{.*}}: 0, {{.*}}0]

// Test html output.
// RUN: llvm-cov show --show-branch-summary --show-branches=count %S/Inputs/branch-c-general.o32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s -format html -o %t.html.dir
// RUN: FileCheck -check-prefix=HTML -input-file=%t.html.dir/coverage/tmp/branch-c-general.c.html %s
// HTML-COUNT-89: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>
// HTML-NOT: Branch (<span class='line-number'><a name='L{{[0-9]+}}' href='#L{{[0-9]+}}'><span>

// RUN: FileCheck -check-prefix HTML-INDEX -input-file %t.html.dir/index.html %s
// HTML-INDEX-LABEL: <table>
// HTML-INDEX: <td class='column-entry-bold'>Filename</td>
// HTML-INDEX: <td class='column-entry-bold'>Function Coverage</td>
// HTML-INDEX: <td class='column-entry-bold'>Line Coverage</td>
// HTML-INDEX: <td class='column-entry-bold'>Region Coverage</td>
// HTML-INDEX: <td class='column-entry-bold'>Branch Coverage</td>
// HTML-INDEX: <a href='coverage{{.*}}branch-c-general.c.html'{{.*}}branch-c-general.c</a>
// HTML-INDEX: <td class='column-entry-green'>
// HTML-INDEX: 100.00% (12/12)
// HTML-INDEX: <td class='column-entry-yellow'>
// HTML-INDEX: 94.44% (221/234)
// HTML-INDEX: <td class='column-entry-yellow'>
// HTML-INDEX: 87.82% (173/197)
// HTML-INDEX: <td class='column-entry-red'>
// HTML-INDEX: 78.16% (136/174)
// HTML-INDEX: <tr class='light-row-bold'>
// HTML-INDEX: Totals
