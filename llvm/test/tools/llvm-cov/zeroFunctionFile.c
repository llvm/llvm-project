#include "Inputs/zeroFunctionFile.h"

int foo(int x) {
  return NOFUNCTIONS(x);
}
int main() {
  return foo(2);
}

// RUN: llvm-profdata merge %S/Inputs/zeroFunctionFile.proftext -o %t.profdata

// RUN: llvm-cov report --respect-coverage-exclusion-markers=false %S/Inputs/zeroFunctionFile.covmapping -instr-profile %t.profdata 2>&1 | FileCheck --check-prefix=REPORT --strict-whitespace %s
// REPORT: Files which contain no functions
// REPORT: zeroFunctionFile.h

// RUN: llvm-cov show --respect-coverage-exclusion-markers=false -j 1 %S/Inputs/zeroFunctionFile.covmapping -format html -instr-profile %t.profdata -o %t.dir
// RUN: FileCheck %s -input-file=%t.dir/index.html -check-prefix=HTML
// HTML-NO: 0.00% (0/0)
// HTML: Files which contain no functions
// HTML: zeroFunctionFile.h
