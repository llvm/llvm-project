// Basic handling of line counts.
// RUN: llvm-profdata merge %S/Inputs/lineExecutionCounts.proftext -o %t.profdata

// before any coverage              // WHOLE-FILE:      | [[@LINE]]|// before
                                    // FILTER-NOT:      | [[@LINE-1]]|// before
int main() {                              // TEXT:   161| [[@LINE]]|int main(
  int x = 0;                              // TEXT:   161| [[@LINE]]|  int x
                                          // TEXT:   161| [[@LINE]]|
  if (x) {                                // TEXT:     0| [[@LINE]]|  if (x)
    x = 0;                                // TEXT:     0| [[@LINE]]|    x = 0
  } else {                                // TEXT:   161| [[@LINE]]|  } else
    x = 1;                                // TEXT:   161| [[@LINE]]|    x = 1
  }                                       // TEXT:   161| [[@LINE]]|  }
                                          // TEXT:   161| [[@LINE]]|
  for (int i = 0; i < 100; ++i) {         // TEXT: 16.2k| [[@LINE]]|  for (
    x = 1;                                // TEXT: 16.1k| [[@LINE]]|    x = 1
  }                                       // TEXT: 16.1k| [[@LINE]]|  }
                                          // TEXT:   161| [[@LINE]]|
  x = x < 10 ? x + 1 : x - 1;             // TEXT:   161| [[@LINE]]|  x =
  x = x > 10 ?                            // TEXT:   161| [[@LINE]]|  x =
        x - 1:                            // TEXT:     0| [[@LINE]]|        x
        x + 1;                            // TEXT:   161| [[@LINE]]|        x
                                          // TEXT:   161| [[@LINE]]|
  return 0;                               // TEXT:   161| [[@LINE]]|  return
}                                         // TEXT:   161| [[@LINE]]|}
// after coverage                   // WHOLE-FILE:      | [[@LINE]]|// after
                                    // FILTER-NOT:      | [[@LINE-1]]|// after

// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -filename-equivalence %s | FileCheck -check-prefixes=TEXT,WHOLE-FILE %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %t.profdata -filename-equivalence -name=main %s | FileCheck -check-prefixes=TEXT,FILTER %s

// Test -output-dir.
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -o %t.dir -instr-profile %t.profdata -filename-equivalence %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -output-dir %t.dir -instr-profile %t.profdata -filename-equivalence -name=main %s
// RUN: FileCheck -check-prefixes=TEXT,WHOLE-FILE -input-file %t.dir/coverage/tmp/showLineExecutionCounts.cpp.txt %s
// RUN: FileCheck -check-prefixes=TEXT,FILTER -input-file %t.dir/functions.txt %s
//
// Test index creation.
// RUN: FileCheck -check-prefix=INDEX -input-file %t.dir/index.txt %s
// INDEX: showLineExecutionCounts.cpp.txt
//
// Test html output.
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -format html -o %t.html.dir -instr-profile %t.profdata -filename-equivalence %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -format html -o %t.html.dir -instr-profile %t.profdata -filename-equivalence -name=main %s
// RUN: FileCheck -check-prefixes=HTML,HTML-WHOLE-FILE -input-file %t.html.dir/coverage/tmp/showLineExecutionCounts.cpp.html %s
// RUN: FileCheck -check-prefixes=HTML,HTML-FILTER -input-file %t.html.dir/functions.html %s
//
// HTML-WHOLE-FILE: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>// before
// HTML-FILTER-NOT: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-45]]</pre></td><td class='code'><pre>// before
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>int main() {
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  int x = 0
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>
// HTML: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre><span class='red'>  if (x) {
// HTML: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre><span class='red'>  }</span>
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>    x = 1;
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  }
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>
// HTML: <td class='covered-line'><pre>16.2k</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  for (int i = 0; i &lt; 100; ++i)
// HTML: <td class='covered-line'><pre>16.1k</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>    x = 1;
// HTML: <td class='covered-line'><pre>16.1k</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  }
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  x = x &lt; 10
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  x = x &gt; 10
// HTML: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre><span class='red'>        x - 1:
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>        x + 1;
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  return 0;
// HTML: <td class='covered-line'><pre>161</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>}
// HTML-WHOLE-FILE: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>// after
// HTML-FILTER-NOT: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-45]]</pre></td><td class='code'><pre>// after
