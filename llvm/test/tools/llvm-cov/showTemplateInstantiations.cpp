// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence %s | FileCheck -check-prefixes=SHARED,ALL %s
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence -name=_Z4funcIbEiT_ %s | FileCheck -check-prefixes=SHARED,FILTER %s

// before coverage   // ALL:          | [[@LINE]]|// before
                     // FILTER-NOT:   | [[@LINE-1]]|// before
template<typename T> // ALL:          | [[@LINE]]|template<typename T>
int func(T x) {      // ALL-NEXT:    2| [[@LINE]]|int func(T x) {
  if(x)              // ALL-NEXT:    2| [[@LINE]]|  if(x)
    return 0;        // ALL-NEXT:    1| [[@LINE]]|    return 0;
  else               // ALL-NEXT:    2| [[@LINE]]|  else
    return 1;        // ALL-NEXT:    1| [[@LINE]]|    return 1;
  int j = 1;         // ALL-NEXT:    0| [[@LINE]]|  int j = 1;
}                    // ALL-NEXT:    2| [[@LINE]]|}

                     // SHARED:       {{^ *(\| )?}}_Z4funcIbEiT_:
                     // SHARED-NEXT:  1| [[@LINE-9]]|int func(T x) {
                     // SHARED-NEXT:  1| [[@LINE-9]]|  if(x)
                     // SHARED-NEXT:  1| [[@LINE-9]]|    return 0;
                     // SHARED-NEXT:  1| [[@LINE-9]]|  else
                     // SHARED-NEXT:  0| [[@LINE-9]]|    return 1;
                     // SHARED-NEXT:  0| [[@LINE-9]]|  int j = 1;
                     // SHARED-NEXT:  1| [[@LINE-9]]|}

                     // ALL:         {{^ *}}| _Z4funcIiEiT_:
                     // FILTER-NOT:  {{^ *(\| )?}} _Z4funcIiEiT_:
                     // ALL-NEXT:    1| [[@LINE-19]]|int func(T x) {
                     // ALL-NEXT:    1| [[@LINE-19]]|  if(x)
                     // ALL-NEXT:    0| [[@LINE-19]]|    return 0;
                     // ALL-NEXT:    1| [[@LINE-19]]|  else
                     // ALL-NEXT:    1| [[@LINE-19]]|    return 1;
                     // ALL-NEXT:    0| [[@LINE-19]]|  int j = 1;
                     // ALL-NEXT:    1| [[@LINE-19]]|}

int main() {         // ALL:         1| [[@LINE]]|int main() {
  func<int>(0);      // ALL-NEXT:    1| [[@LINE]]|  func<int>(0);
  func<bool>(true);  // ALL-NEXT:    1| [[@LINE]]|  func<bool>(true);
  return 0;          // ALL-NEXT:    1| [[@LINE]]|  return 0;
}                    // ALL-NEXT:    1| [[@LINE]]|}
// after coverage    // ALL-NEXT:     | [[@LINE]]|// after
                     // FILTER-NOT:   | [[@LINE-1]]|// after

// Test html output.
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence %s -format html -o %t.html.dir
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence -name=_Z4funcIbEiT_ %s -format html -o %t.html.dir
// RUN: FileCheck -check-prefixes=HTML-SHARED,HTML-ALL -input-file=%t.html.dir/coverage/tmp/showTemplateInstantiations.cpp.html %s
// RUN: FileCheck -check-prefixes=HTML-SHARED,HTML-FILTER -input-file=%t.html.dir/functions.html %s

// HTML-ALL: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>// before
// HTML-FILTER-NOT: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-45]]</pre></td><td class='code'><pre>// before
// HTML-ALL: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>template&lt;typename T&gt;
// HTML-ALL: <td class='covered-line'><pre>2</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>int func(T x) {
// HTML-ALL: <td class='covered-line'><pre>2</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  if(x)
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>    ret
// HTML-ALL: <td class='covered-line'><pre>2</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  else
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>    ret
// HTML-ALL: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>
// HTML-ALL: <td class='covered-line'><pre>2</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>}

// HTML-SHARED: <div class='source-name-title'><pre>_Z4funcIbEiT_</pre></div><table>
// HTML-SHARED: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-53]]</pre></td><td class='code'><pre>int func(T x) {
// HTML-SHARED: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-53]]</pre></td><td class='code'><pre>  if(x)
// HTML-SHARED: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-53]]</pre></td><td class='code'><pre>    ret
// HTML-SHARED: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-53]]</pre></td><td class='code'><pre>  else
// HTML-SHARED: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-53]]</pre></td><td class='code'><pre>
// HTML-SHARED: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-53]]</pre></td><td class='code'><pre>
// HTML-SHARED: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-53]]</pre></td><td class='code'><pre>}

// HTML-ALL: <div class='source-name-title'><pre>_Z4funcIiEiT_</pre></div><table>
// HTML-FILTER-NOT: <div class='source-name-title'><pre>_Z4funcIiEiT_</pre></div><table>
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-63]]</pre></td><td class='code'><pre>int func(T x) {
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-63]]</pre></td><td class='code'><pre>  if(x)
// HTML-ALL: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-63]]</pre></td><td class='code'><pre>
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-63]]</pre></td><td class='code'><pre>  else
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-63]]</pre></td><td class='code'><pre>    ret
// HTML-ALL: <td class='uncovered-line'><pre>0</pre></td><td class='line-number'><pre>[[@LINE-63]]</pre></td><td class='code'><pre>
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-63]]</pre></td><td class='code'><pre>}

// HTML-ALL: td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>int main() {
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  func&lt;int&gt;(0);
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  func&lt;bool&gt;(true);
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>  return 0;
// HTML-ALL: <td class='covered-line'><pre>1</pre></td><td class='line-number'><pre>[[@LINE-44]]</pre></td><td class='code'><pre>}

// HTML-ALL: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-45]]</pre></td><td class='code'><pre>// after
// HTML-FILTER-NOT: <td class='uncovered-line'></td><td class='line-number'><pre>[[@LINE-46]]</pre></td><td class='code'><pre>// after
