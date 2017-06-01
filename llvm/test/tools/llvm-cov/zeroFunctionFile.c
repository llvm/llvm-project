#include "Inputs/zeroFunctionFile.h"

int foo(int x) {
  return NOFUNCTIONS(x);
}
int main() {
  return foo(2);
}

// RUN: llvm-profdata merge %S/Inputs/zeroFunctionFile.proftext -o %t.profdata

// RUN: llvm-cov report %S/Inputs/zeroFunctionFile.covmapping -instr-profile %t.profdata 2>&1 | FileCheck --check-prefix=REPORT --strict-whitespace %s
// REPORT: 0                 0         -           0                 0         -               0               0         -           0                 0         -
// REPORT-NO: 0%

// RUN: llvm-cov show %S/Inputs/zeroFunctionFile.covmapping -format html -instr-profile %t.profdata -o %t.dir
// RUN: FileCheck %s -input-file=%t.dir/index.html -check-prefix=HTML
// HTML: <td class='column-entry-green'><pre>- (0/0)
// HTML-NO: 0.00% (0/0)
