// Basic handling of line counts.
// RUN: llvm-profdata merge %S/Inputs/lineExecutionCounts.proftext -o %t.profdata

// before any coverage              // WHOLE-FILE:      | [[@LINE]]|// before
                                    // FILTER-NOT:      | [[@LINE-1]]|// before
int main() {                             // CHECK:   161| [[@LINE]]|int main(
  int x = 0;                             // CHECK:   161| [[@LINE]]|  int x
                                         // CHECK:   161| [[@LINE]]|
  if (x) {                               // CHECK:     0| [[@LINE]]|  if (x)
    x = 0;                               // CHECK:     0| [[@LINE]]|    x = 0
  } else {                               // CHECK:   161| [[@LINE]]|  } else
    x = 1;                               // CHECK:   161| [[@LINE]]|    x = 1
  }                                      // CHECK:   161| [[@LINE]]|  }
                                         // CHECK:   161| [[@LINE]]|
  for (int i = 0; i < 100; ++i) {        // CHECK: 16.2k| [[@LINE]]|  for (
    x = 1;                               // CHECK: 16.1k| [[@LINE]]|    x = 1
  }                                      // CHECK: 16.1k| [[@LINE]]|  }
                                         // CHECK:   161| [[@LINE]]|
  x = x < 10 ? x + 1 : x - 1;            // CHECK:   161| [[@LINE]]|  x =
  x = x > 10 ?                           // CHECK:   161| [[@LINE]]|  x =
        x - 1:                           // CHECK:     0| [[@LINE]]|        x
        x + 1;                           // CHECK:   161| [[@LINE]]|        x
                                         // CHECK:   161| [[@LINE]]|
  return 0;                              // CHECK:   161| [[@LINE]]|  return
}                                        // CHECK:   161| [[@LINE]]|}
// after coverage                   // WHOLE-FILE:      | [[@LINE]]|// after
                                    // FILTER-NOT:      | [[@LINE-1]]|// after

// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -filename-equivalence %s | FileCheck -check-prefixes=CHECK,WHOLE-FILE %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -filename-equivalence -name=main %s | FileCheck -check-prefixes=CHECK,FILTER %s

// Test -output-dir.
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -o %t.dir -instr-profile %t.profdata -filename-equivalence %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -output-dir %t.dir -instr-profile %t.profdata -filename-equivalence -name=main %s
// RUN: FileCheck -check-prefixes=CHECK,WHOLE-FILE -input-file %t.dir/coverage/tmp/showLineExecutionCounts.cpp.txt %s
// RUN: FileCheck -check-prefixes=CHECK,FILTER -input-file %t.dir/functions.txt %s
//
// Test index creation.
// RUN: FileCheck -check-prefix=INDEX -input-file %t.dir/index.txt %s
// INDEX: showLineExecutionCounts.cpp.txt
