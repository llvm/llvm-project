// Basic handling of line counts.
// RUN: llvm-profdata merge %S/Inputs/lineExecutionCounts.proftext -o %t.profdata

// before any coverage              // WHOLE-FILE: [[@LINE]]|      |// before
                                    // FILTER-NOT: [[@LINE-1]]|    |// before
int main() {                              // TEXT: [[@LINE]]|   161|int main(
  int x = 0;                              // TEXT: [[@LINE]]|   161|  int x
                                          // TEXT: [[@LINE]]|   161|
  if (x) {                                // TEXT: [[@LINE]]|     0|  if (x)
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

// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -filename-equivalence %s | FileCheck -check-prefixes=TEXT,WHOLE-FILE %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -filename-equivalence -name=main %s | FileCheck -check-prefixes=TEXT,FILTER %s

// Test -output-dir.
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -o %t.dir -instr-profile %t.profdata -filename-equivalence %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -output-dir %t.dir -instr-profile %t.profdata -filename-equivalence -name=main %s
// RUN: FileCheck -check-prefixes=TEXT,WHOLE-FILE -input-file %t.dir/coverage/tmp/showLineExecutionCounts.cpp.txt %s
// RUN: FileCheck -check-prefixes=TEXT,FILTER -input-file %t.dir/functions.txt %s
//
// RUN: llvm-cov export %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -name=main 2>/dev/null > %t.export.json
// RUN: FileCheck -input-file %t.export.json %S/Inputs/lineExecutionCounts.json
// RUN: cat %t.export.json | %python -c "import json, sys; json.loads(sys.stdin.read())"
//
// Test html output.
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -format html -o %t.html.dir -instr-profile %t.profdata -filename-equivalence %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -format html -o %t.html.dir -instr-profile %t.profdata -filename-equivalence -name=main %s
// RUN: FileCheck -check-prefixes=HTML,HTML-WHOLE-FILE -input-file %t.html.dir/coverage/tmp/showLineExecutionCounts.cpp.html %s
// RUN: FileCheck -check-prefixes=HTML,HTML-FILTER -input-file %t.html.dir/functions.html %s
//
// HTML-WHOLE-FILE: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// before
// HTML-FILTER-NOT: <td class='line-number'><a name='L[[@LINE-45]]' href='#L[[@LINE-45]]'><pre>[[@LINE-45]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// before
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>int main() {
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>  int x = 0
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>  if (x) {
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre><span class='red'>  }</span>
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>    x = 1;
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>  }
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>16.2k</pre></td><td class='code'><pre>  for (int i = 0; i &lt; 100; ++i)
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>16.1k</pre></td><td class='code'><pre>    x = 1;
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>16.1k</pre></td><td class='code'><pre>  }
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>  x = x &lt; 10
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>  x = x &gt; 10
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>        x - 1:
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>        x + 1;
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>  return 0;
// HTML: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>}
// HTML-WHOLE-FILE: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// after
// HTML-FILTER-NOT: <td class='line-number'><a name='L[[@LINE-45]]' href='#L[[@LINE-45]]'><pre>[[@LINE-45]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// after
//
// Test index creation.
// RUN: FileCheck -check-prefix=TEXT-INDEX -input-file %t.dir/index.txt %s
// TEXT-INDEX: Filename
// TEXT-INDEX-NEXT: ---
// TEXT-INDEX-NEXT: {{.*}}showLineExecutionCounts.cpp
//
// RUN: FileCheck -check-prefix HTML-INDEX -input-file %t.html.dir/index.html %s
// HTML-INDEX-LABEL: <table>
// HTML-INDEX: <td class='column-entry-left'>Filename</td>
// HTML-INDEX: <td class='column-entry'>Function Coverage</td>
// HTML-INDEX: <td class='column-entry'>Instantiation Coverage</td>
// HTML-INDEX: <td class='column-entry'>Line Coverage</td>
// HTML-INDEX: <td class='column-entry'>Region Coverage</td>
// HTML-INDEX: <a href='coverage{{.*}}showLineExecutionCounts.cpp.html'{{.*}}showLineExecutionCounts.cpp</a>
// HTML-INDEX: <td class='column-entry-green'>
// HTML-INDEX: 100.00% (1/1)
// HTML-INDEX: <td class='column-entry-green'>
// HTML-INDEX: 100.00% (1/1)
// HTML-INDEX: <td class='column-entry-yellow'>
// HTML-INDEX: 80.00% (16/20)
// HTML-INDEX: <td class='column-entry-red'>
// HTML-INDEX: 70.00% (7/10)
// HTML-INDEX: TOTALS
