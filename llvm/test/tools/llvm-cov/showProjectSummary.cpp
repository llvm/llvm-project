// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/showProjectSummary.proftext

int main(int argc, char ** argv) {
  int x=0;
  for (int i = 0; i < 20; ++i)
    x *= 2;
  if (x >= 100)
    x = x / 2;
  else
    x = x * 2;
  return x;
}

// Test console output.
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck -check-prefixes=TEXT,TEXT-FILE,TEXT-HEADER %S/Inputs/showProjectSummary.test
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile %t.profdata -path-equivalence=/tmp,%S -name=main %s | FileCheck -check-prefixes=TEXT,TEXT-FILE,TEXT-HEADER %S/Inputs/showProjectSummary.test
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile %t.profdata -project-title "Test Suite" -path-equivalence=/tmp,%S %s | FileCheck -check-prefixes=TEXT-TITLE,TEXT,TEXT-FILE,TEXT-HEADER %S/Inputs/showProjectSummary.test
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile %t.profdata -project-title "Test Suite" -name=main -path-equivalence=/tmp,%S %s | FileCheck -check-prefixes=TEXT-FUNCTION,TEXT-HEADER %S/Inputs/showProjectSummary.test
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile=%t.profdata -o %t.dir -path-equivalence=/tmp,%S %s
// RUN: FileCheck -check-prefixes=TEXT-FOOTER -input-file=%t.dir/index.txt %S/Inputs/showProjectSummary.test

// Test html output.
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -format=html -o %t.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S %s
// RUN: FileCheck -check-prefixes=HTML,HTML-FILE,HTML-HEADER -input-file %t.dir/coverage/tmp/showProjectSummary.cpp.html %S/Inputs/showProjectSummary.test
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -format=html -o %t.dir -instr-profile %t.profdata -project-title "Test Suite" -path-equivalence=/tmp,%S %s
// RUN: FileCheck -check-prefixes=HTML-TITLE,HTML,HTML-FILE,HTML-HEADER -input-file %t.dir/coverage/tmp/showProjectSummary.cpp.html %S/Inputs/showProjectSummary.test
// RUN: FileCheck -check-prefixes=HTML-TITLE,HTML,HTML-FOOTER -input-file %t.dir/index.html %S/Inputs/showProjectSummary.test
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -format=html -o %t.filtered.dir -instr-profile %t.profdata  -project-title "Test Suite" -path-equivalence=/tmp,%S -name=main %s
// RUN: FileCheck -check-prefixes=HTML-TITLE,HTML,HTML-FOOTER -input-file %t.filtered.dir/index.html %S/Inputs/showProjectSummary.test
