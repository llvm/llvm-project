// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence %s | FileCheck -check-prefix=CHECK -check-prefix=ALL %s
// RUN: llvm-cov show %S/Inputs/templateInstantiations.covmapping -instr-profile %S/Inputs/templateInstantiations.profdata -filename-equivalence -name=_Z4funcIbEiT_ %s | FileCheck -check-prefix=CHECK -check-prefix=FILTER %s

// before coverage   // WHOLE-FILE:   | [[@LINE]]|// before
                     // FILTER-NOT:   | [[@LINE-1]]|// before
template<typename T> // ALL:          | [[@LINE]]|template<typename T>
int func(T x) {      // ALL-NEXT:    2| [[@LINE]]|int func(T x) {
  if(x)              // ALL-NEXT:    2| [[@LINE]]|  if(x)
    return 0;        // ALL-NEXT:    1| [[@LINE]]|    return 0;
  else               // ALL-NEXT:    1| [[@LINE]]|  else
    return 1;        // ALL-NEXT:    1| [[@LINE]]|    return 1;
  int j = 1;         // ALL-NEXT:    0| [[@LINE]]|  int j = 1;
}                    // ALL-NEXT:    1| [[@LINE]]|}

                     // CHECK:       {{^ *(\| )?}}_Z4funcIbEiT_:
                     // CHECK-NEXT:  1| [[@LINE-9]]|int func(T x) {
                     // CHECK-NEXT:  1| [[@LINE-9]]|  if(x)
                     // CHECK-NEXT:  1| [[@LINE-9]]|    return 0;
                     // CHECK-NEXT:  1| [[@LINE-9]]|  else
                     // CHECK-NEXT:  0| [[@LINE-9]]|    return 1;
                     // CHECK-NEXT:  0| [[@LINE-9]]|  int j = 1;
                     // CHECK-NEXT:  1| [[@LINE-9]]|}

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
