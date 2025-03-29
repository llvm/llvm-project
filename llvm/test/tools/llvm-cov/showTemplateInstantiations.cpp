// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -path-equivalence=/tmp,%S %s | FileCheck -check-prefixes=SHARED,ALL %s
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -path-equivalence=/tmp,%S -name=_Z4funcIbEiT_ %s | FileCheck -check-prefixes=SHARED,FILTER %s

// before coverage   // ALL:         [[@LINE]]|  |// before
                     // FILTER-NOT:[[@LINE-1]]|  |// before
template<typename T> // ALL:         [[@LINE]]|  |template<typename T>
int func(T x) {      // ALL-NEXT:    [[@LINE]]| 2|int func(T x) {
  if(x)              // ALL-NEXT:    [[@LINE]]| 2|  if(x)
    return 0;        // ALL-NEXT:    [[@LINE]]| 1|    return 0;
  else               // ALL-NEXT:    [[@LINE]]| 1|  else
    return 1;        // ALL-NEXT:    [[@LINE]]| 1|    return 1;
  int j = 1;         // ALL-NEXT:    [[@LINE]]| 0|  int j = 1;
}                    // ALL-NEXT:    [[@LINE]]| 0|}

                     // ALL:         _Z4{{[a-z]+}}IiEiT_:
                     // FILTER-NOT:  _Z4{{[a-z]+}}IiEiT_:
                     // ALL:         [[@LINE-10]]| 1|int func(T x) {
                     // ALL-NEXT:    [[@LINE-10]]| 1|  if(x)
                     // ALL-NEXT:    [[@LINE-10]]| 0|    return 0;
                     // ALL-NEXT:    [[@LINE-10]]| 1|  else
                     // ALL-NEXT:    [[@LINE-10]]| 1|    return 1;
                     // ALL-NEXT:    [[@LINE-10]]| 0|  int j = 1;
                     // ALL-NEXT:    [[@LINE-10]]| 0|}

                     // SHARED:       {{^ *(\| )?}}_Z4funcIbEiT_:
                     // SHARED:       [[@LINE-19]]| 1|int func(T x) {
                     // SHARED-NEXT:  [[@LINE-19]]| 1|  if(x)
                     // SHARED-NEXT:  [[@LINE-19]]| 1|    return 0;
                     // SHARED-NEXT:  [[@LINE-19]]| 0|  else
                     // SHARED-NEXT:  [[@LINE-19]]| 0|    return 1;
                     // SHARED-NEXT:  [[@LINE-19]]| 0|  int j = 1;
                     // SHARED-NEXT:  [[@LINE-19]]| 0|}

int main() {         // ALL:         [[@LINE]]| 1|int main() {
  func<int>(0);      // ALL-NEXT:    [[@LINE]]| 1|  func<int>(0);
  func<bool>(true);  // ALL-NEXT:    [[@LINE]]| 1|  func<bool>(true);
  return 0;          // ALL-NEXT:    [[@LINE]]| 1|  return 0;
}                    // ALL-NEXT:    [[@LINE]]| 1|}
// after coverage    // ALL-NEXT:    [[@LINE]]|  |// after
                     // FILTER-NOT:[[@LINE-1]]|  |// after

// Test html output.
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -path-equivalence=/tmp,%S %s -format html -o %t.html.dir
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -path-equivalence=/tmp,%S -name=_Z4funcIbEiT_ %s -format html -o %t.html.filtered.dir
// RUN: FileCheck -check-prefixes=HTML-SHARED,HTML-ALL -input-file=%t.html.dir/coverage/tmp/showTemplateInstantiations.cpp.html %s
// RUN: FileCheck -check-prefixes=HTML-SHARED,HTML-FILTER -input-file=%t.html.filtered.dir/coverage/tmp/showTemplateInstantiations.cpp.html %s

// HTML-ALL: <td class='line-number'><a name='L4' href='#L4'><pre>4</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// before
// HTML-FILTER-NOT: <td class='line-number'><a name='L4' href='#L4'><pre>4</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// before
// HTML-ALL: <td class='line-number'><a name='L6' href='#L6'><pre>6</pre></a></td><td class='skipped-line'></td><td class='code'><pre>template&lt;typename T&gt;

// HTML-ALL: <div class='source-name-title'><pre>_Z4funcIiEiT_</pre></div>
// HTML-FILTER-NOT: <div class='source-name-title'><pre>_Z4funcIiEiT_</pre></div><table>
// HTML-ALL: <td class='line-number'><a name='L7' href='#L7'><pre>7</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>int func(T x) {

// HTML-SHARED: <div class='source-name-title'><pre>_Z4funcIbEiT_</pre></div>
// HTML-SHARED: <td class='line-number'><a name='L7' href='#L7'><pre>7</pre></a></td><td class='covered-line'><pre>1</pre></td><td class='code'><pre>int func(T x) {

// RUN: FileCheck -check-prefix=HTML-JUMP -input-file=%t.html.dir/coverage/tmp/showTemplateInstantiations.cpp.html %s
// HTML-JUMP: <a href='javascript:next_line()'>next uncovered line (L)</a>
// HTML-JUMP-NOT: <a href='javascript:next_line()'>next uncovered line (L)</a>
// HTML-JUMP: <a href='javascript:next_region()'>next uncovered region (R)</a>
// HTML-JUMP-NOT: <a href='javascript:next_region()'>next uncovered region (R)</a>
// HTML-JUMP: <a href='javascript:next_branch()'>next uncovered branch (B)</a>
// HTML-JUMP-NOT: <a href='javascript:next_branch()'>next uncovered branch (B)</a>

// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -show-instantiations=false -path-equivalence=/tmp,%S %s | FileCheck -check-prefix=NO_INSTS %s
// NO_INSTS-NOT: {{^ *}}| _Z4funcIbEiT_:
// NO_INSTS-NOT: {{^ *}}| _Z4funcIiEiT_:

// RUN: llvm-cov report %S/Inputs/templateInstantiations.covmapping -dump -instr-profile %S/Inputs/templateInstantiations.profdata -path-equivalence=/tmp,%S %s | FileCheck -check-prefix=DUMP %s
// DUMP: InstantiationGroup: Definition at line 7, column 15 with size = 2
// DUMP: InstantiationGroup: main with size = 1
