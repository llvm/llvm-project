// Basic handling of line counts.
// RUN: llvm-profdata merge %S/Inputs/lineExecutionCounts.proftext -o %t.profdata

// before any coverage              // WHOLE-FILE: [[@LINE]]|      |// before
                                    // FILTER-NOT: [[@LINE-1]]|    |// before
int main() {                              // TEXT: [[@LINE]]|   161|int main(
  int x = 0;                              // TEXT: [[@LINE]]|   161|  int x
                                          // TEXT: [[@LINE]]|   161|
  if (x) {                                // TEXT: [[@LINE]]|   161|  if (x)
    x = 0;                                // TEXT: [[@LINE]]|     0|    x = 0
  } else {                                // TEXT: [[@LINE]]|   161|  } else
    x = 1;                                // TEXT: [[@LINE]]|   161|    x = 1
  }                                       // TEXT: [[@LINE]]|   161|  }
                                          // TEXT: [[@LINE]]|   161|
  for (int i = 0; i < 100; ++i) {         // TEXT: [[@LINE]]| 16.2k|  for (
    x = 1;                                // TEXT: [[@LINE]]| 16.1k|    x = 1
  }                                       // TEXT: [[@LINE]]| 16.1k|  }
                                          // TEXT: [[@LINE]]|   161|
  x = x < 10 ? x + 1 : x - 1;             // TEXT: [[@LINE]]|   161|  x =
  x = x > 10 ?                            // TEXT: [[@LINE]]|   161|  x =
        x - 1:                            // TEXT: [[@LINE]]|     0|        x
        x + 1;                            // TEXT: [[@LINE]]|   161|        x
                                          // TEXT: [[@LINE]]|   161|
  return 0;                               // TEXT: [[@LINE]]|   161|  return
}                                         // TEXT: [[@LINE]]|   161|}
// after coverage                   // WHOLE-FILE: [[@LINE]]|      |// after
                                    // FILTER-NOT: [[@LINE-1]]|    |// after

// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck -check-prefixes=TEXT,WHOLE-FILE %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -path-equivalence=/tmp,%S -name=main %s | FileCheck -check-prefixes=TEXT,FILTER %s

// Test -output-dir.
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -o %t.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -output-dir %t.filtered.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S -name=main %s
// RUN: FileCheck -check-prefixes=TEXT,WHOLE-FILE -input-file %t.dir/coverage/tmp/showLineExecutionCounts.cpp.txt %s
// RUN: FileCheck -check-prefixes=TEXT,FILTER -input-file %t.filtered.dir/coverage/tmp/showLineExecutionCounts.cpp.txt %s
//
// RUN: llvm-cov export %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata 2>/dev/null -summary-only > %t.export-summary.json
// RUN: not grep '"name":"main"' %t.export-summary.json
//
// Test html output.
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -format html -o %t.html.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -format html -o %t.html.filtered.dir -instr-profile %t.profdata -path-equivalence=/tmp,%S -name=main %s
// RUN: FileCheck -check-prefixes=HTML,HTML-WHOLE-FILE -input-file %t.html.dir/coverage/tmp/showLineExecutionCounts.cpp.html %s
// RUN: FileCheck -check-prefixes=HTML,HTML-FILTER -input-file %t.html.filtered.dir/coverage/tmp/showLineExecutionCounts.cpp.html %s
//
// HTML-WHOLE-FILE: <td class='line-number'><a name='L4' href='#L4'><pre>4</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// before
// HTML-FILTER-NOT: <td class='line-number'><a name='L4' href='#L4'><pre>4</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// before
// HTML: <td class='line-number'><a name='L6' href='#L6'><pre>6</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>int main() {
// HTML-WHOLE-FILE: <td class='line-number'><a name='L26' href='#L26'><pre>26</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// after
// HTML-FILTER-NOT: <td class='line-number'><a name='L26' href='#L26'><pre>26</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// after
//
// Test index creation.
// RUN: FileCheck -check-prefix=TEXT-INDEX -input-file %t.dir/index.txt %s
// TEXT-INDEX: Filename
// TEXT-INDEX-NEXT: ---
// TEXT-INDEX-NEXT: {{.*}}showLineExecutionCounts.cpp
//
// RUN: FileCheck -check-prefix HTML-INDEX -input-file %t.html.dir/index.html %s
// HTML-INDEX-LABEL: <table>
// HTML-INDEX: <td class='column-entry-bold'>Filename</td>
// HTML-INDEX: <td class='column-entry-bold'>Function Coverage</td>
// HTML-INDEX: <td class='column-entry-bold'>Line Coverage</td>
// HTML-INDEX: <td class='column-entry-bold'>Region Coverage</td>
// HTML-INDEX: <a href='coverage{{.*}}showLineExecutionCounts.cpp.html'{{.*}}showLineExecutionCounts.cpp</a>
// HTML-INDEX: <td class='column-entry-green'>
// HTML-INDEX: 100.00% (1/1)
// HTML-INDEX: <td class='column-entry-yellow'>
// HTML-INDEX: 90.00% (18/20)
// HTML-INDEX: <td class='column-entry-red'>
// HTML-INDEX: 72.73% (8/11)
// HTML-INDEX: <tr class='light-row-bold'>
// HTML-INDEX: Totals
