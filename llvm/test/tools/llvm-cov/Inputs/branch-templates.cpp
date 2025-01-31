




#include <stdio.h>
template<typename T>
void unused(T x) {
  return;
}

template<typename T>
int func(T x) {
  if(x)       // BRCOV: |  Branch ([[@LINE]]:6): [True: 0, False: 1]
    return 0; // BRCOV: |  Branch ([[@LINE-1]]:6): [True: 1, False: 0]
  else        // BRCOV: |  Branch ([[@LINE-2]]:6): [True: 0, False: 1]
    return 1;
  int j = 1;
}

              // CHECK-LABEL: _Z4funcIiEiT_:
              // BRCOV: |  |  Branch ([[@LINE-8]]:6): [True: 0, False: 1]
              // CHECK-LABEL: _Z4funcIbEiT_:
              // BRCOV: |  |  Branch ([[@LINE-10]]:6): [True: 1, False: 0]
              // CHECK-LABEL: _Z4funcIfEiT_:
              // BRCOV: |  |  Branch ([[@LINE-12]]:6): [True: 0, False: 1]


int main() {
  if (func<int>(0))      // BRCOV: |  Branch ([[@LINE]]:7): [True: 1, False: 0]
    printf("case1\n");
  if (func<bool>(true))  // BRCOV: |  Branch ([[@LINE]]:7): [True: 0, False: 1]
    printf("case2\n");
  if (func<float>(0.0))  // BRCOV: |  Branch ([[@LINE]]:7): [True: 1, False: 0]
    printf("case3\n");
  (void)0;
  return 0;
}
