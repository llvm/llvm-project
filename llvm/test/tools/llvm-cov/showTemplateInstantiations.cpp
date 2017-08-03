// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence %s | FileCheck -check-prefixes=SHARED,ALL %s
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence -name=_Z4funcIbEiT_ %s | FileCheck -check-prefixes=SHARED,FILTER %s

// before coverage   // ALL:         [[@LINE]]|  |// before
                     // FILTER-NOT:[[@LINE-1]]|  |// before
template<typename T> // ALL:         [[@LINE]]|  |template<typename T>
int func(T x) {      // ALL-NEXT:    [[@LINE]]| 2|int func(T x) {
  if(x)              // ALL-NEXT:    [[@LINE]]| 2|  if(x)
    return 0;        // ALL-NEXT:    [[@LINE]]| 1|    return 0;
  else               // ALL-NEXT:    [[@LINE]]| 2|  else
    return 1;        // ALL-NEXT:    [[@LINE]]| 1|    return 1;
  int j = 1;         // ALL-NEXT:    [[@LINE]]| 0|  int j = 1;
}                    // ALL-NEXT:    [[@LINE]]| 2|}

                     // SHARED:       {{^ *(\| )?}}_Z4funcIbEiT_:
                     // SHARED:       [[@LINE-9]]| 1|int func(T x) {
                     // SHARED-NEXT:  [[@LINE-9]]| 1|  if(x)
                     // SHARED-NEXT:  [[@LINE-9]]| 1|    return 0;
                     // SHARED-NEXT:  [[@LINE-9]]| 1|  else
                     // SHARED-NEXT:  [[@LINE-9]]| 0|    return 1;
                     // SHARED-NEXT:  [[@LINE-9]]| 0|  int j = 1;
                     // SHARED-NEXT:  [[@LINE-9]]| 1|}

                     // ALL:         {{^ *}}| _Z4funcIiEiT_:
                     // FILTER-NOT:  {{^ *(\| )?}} _Z4funcIiEiT_:
                     // ALL:         [[@LINE-19]]| 1|int func(T x) {
                     // ALL-NEXT:    [[@LINE-19]]| 1|  if(x)
                     // ALL-NEXT:    [[@LINE-19]]| 0|    return 0;
                     // ALL-NEXT:    [[@LINE-19]]| 1|  else
                     // ALL-NEXT:    [[@LINE-19]]| 1|    return 1;
                     // ALL-NEXT:    [[@LINE-19]]| 0|  int j = 1;
                     // ALL-NEXT:    [[@LINE-19]]| 1|}

int main() {         // ALL:         [[@LINE]]| 1|int main() {
  func<int>(0);      // ALL-NEXT:    [[@LINE]]| 1|  func<int>(0);
  func<bool>(true);  // ALL-NEXT:    [[@LINE]]| 1|  func<bool>(true);
  return 0;          // ALL-NEXT:    [[@LINE]]| 1|  return 0;
}                    // ALL-NEXT:    [[@LINE]]| 1|}
// after coverage    // ALL-NEXT:    [[@LINE]]|  |// after
                     // FILTER-NOT:[[@LINE-1]]|  |// after

// Test html output.
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence %s -format html -o %t.html.dir
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence -name=_Z4funcIbEiT_ %s -format html -o %t.html.dir
// RUN: FileCheck -check-prefixes=HTML-SHARED,HTML-ALL -input-file=%t.html.dir/coverage/tmp/showTemplateInstantiations.cpp.html %s
// RUN: FileCheck -check-prefixes=HTML-SHARED,HTML-FILTER -input-file=%t.html.dir/functions.html %s

// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// before
// HTML-FILTER-NOT: <td class='line-number'><a name='L[[@LINE-45]]' href='#L[[@LINE-45]]'><pre>[[@LINE-45]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// before
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>template&lt;typename T&gt;
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>2</pre></td><td class='code'><pre>int func(T x) {
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>2</pre></td><td class='code'><pre>  if(x)
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>    ret
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>2</pre></td><td class='code'><pre>  else
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>    ret
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>2</pre></td><td class='code'><pre>}

// HTML-SHARED: <div class='source-name-title'><pre>_Z4funcIbEiT_</pre></div>
// HTML-SHARED: <td class='line-number'><a name='L[[@LINE-53]]' href='#L[[@LINE-53]]'><pre>[[@LINE-53]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>int func(T x) {
// HTML-SHARED: <td class='line-number'><a name='L[[@LINE-53]]' href='#L[[@LINE-53]]'><pre>[[@LINE-53]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>  if(x)
// HTML-SHARED: <td class='line-number'><a name='L[[@LINE-53]]' href='#L[[@LINE-53]]'><pre>[[@LINE-53]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>    ret
// HTML-SHARED: <td class='line-number'><a name='L[[@LINE-53]]' href='#L[[@LINE-53]]'><pre>[[@LINE-53]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>  else
// HTML-SHARED: <td class='line-number'><a name='L[[@LINE-53]]' href='#L[[@LINE-53]]'><pre>[[@LINE-53]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>
// HTML-SHARED: <td class='line-number'><a name='L[[@LINE-53]]' href='#L[[@LINE-53]]'><pre>[[@LINE-53]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>
// HTML-SHARED: <td class='line-number'><a name='L[[@LINE-53]]' href='#L[[@LINE-53]]'><pre>[[@LINE-53]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>}

// HTML-ALL: <div class='source-name-title'><pre>_Z4funcIiEiT_</pre></div>
// HTML-FILTER-NOT: <div class='source-name-title'><pre>_Z4funcIiEiT_</pre></div><table>
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-63]]' href='#L[[@LINE-63]]'><pre>[[@LINE-63]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>int func(T x) {
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-63]]' href='#L[[@LINE-63]]'><pre>[[@LINE-63]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>  if(x)
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-63]]' href='#L[[@LINE-63]]'><pre>[[@LINE-63]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-63]]' href='#L[[@LINE-63]]'><pre>[[@LINE-63]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>  else
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-63]]' href='#L[[@LINE-63]]'><pre>[[@LINE-63]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>    ret
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-63]]' href='#L[[@LINE-63]]'><pre>[[@LINE-63]]</pre></a></td><td class='uncovered-line'><pre>0</pre></td><td class='code'><pre>
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-63]]' href='#L[[@LINE-63]]'><pre>[[@LINE-63]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>}

// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>int main() {
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>  func&lt;int&gt;(0);
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>  func&lt;bool&gt;(true);
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>  return 0;
// HTML-ALL: <td class='line-number'><a name='L[[@LINE-44]]' href='#L[[@LINE-44]]'><pre>[[@LINE-44]]</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>}

// HTML-ALL: <td class='line-number'><a name='L[[@LINE-45]]' href='#L[[@LINE-45]]'><pre>[[@LINE-45]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// after
// HTML-FILTER-NOT: <td class='line-number'><a name='L[[@LINE-46]]' href='#L[[@LINE-46]]'><pre>[[@LINE-46]]</pre></a></td><td class='uncovered-line'></td><td class='code'><pre>// after

// RUN: FileCheck -check-prefix=HTML-JUMP -input-file=%t.html.dir/coverage/tmp/showTemplateInstantiations.cpp.html %s
// HTML-JUMP: <pre>Source (<a href='#L{{[0-9]+}}'>jump to first uncovered line</a>)</pre>
// HTML-JUMP-NOT: <pre>Source (<a href='#L{{[0-9]+}}'>jump to first uncovered line</a>)</pre>
