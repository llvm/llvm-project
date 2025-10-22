



#define COND1 (a == b)
#define COND2 (a != b)
#define COND3 (COND1 && COND2)
#define COND4 (COND3 ? COND2 : COND1) // BRCOV: | Branch ([[@LINE]]:15): [True: 1, False: [[#min(C,2)]]]
#define MACRO1 COND3
#define MACRO2 MACRO1
#define MACRO3 MACRO2

#include <stdlib.h>

// CHECK: |{{ +}}[[#min(C,3)]]|bool func(
bool func(int a, int b) {
  bool c = COND1 && COND2; // BRCOV: |  |  |  Branch ([[@LINE-12]]:15): [True: 1, False: [[#min(C,2)]]]
                           // BRCOV: |  |  |  Branch ([[@LINE-12]]:15): [True: 0, False: 1]
  bool d = COND3;          // BRCOV: |  |  |  |  |  Branch ([[@LINE-14]]:15): [True: 1, False: [[#min(C,2)]]]
                           // BRCOV: |  |  |  |  |  Branch ([[@LINE-14]]:15): [True: 0, False: 1]
  bool e = MACRO1;         // BRCOV: |  |  |  |  |  |  |  Branch ([[@LINE-16]]:15): [True: 1, False: [[#min(C,2)]]]
                           // BRCOV: |  |  |  |  |  |  |  Branch ([[@LINE-16]]:15): [True: 0, False: 1]
  bool f = MACRO2;         // BRCOV: |  |  |  |  |  |  |  |  |  Branch ([[@LINE-18]]:15): [True: 1, False: [[#min(C,2)]]]
                           // BRCOV: |  |  |  |  |  |  |  |  |  Branch ([[@LINE-18]]:15): [True: 0, False: 1]
  bool g = MACRO3;         // BRCOV: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-20]]:15): [True: 1, False: [[#min(C,2)]]]
                           // BRCOV: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-20]]:15): [True: 0, False: 1]
  return c && d && e && f && g;
                           // BRCOV: |  Branch ([[@LINE-1]]:10): [True: 0, False: [[#min(C,3)]]]
                           // BRCOV: |  Branch ([[@LINE-2]]:15): [True: 0, False: 0]
                           // BRCOV: |  Branch ([[@LINE-3]]:20): [True: 0, False: 0]
                           // BRCOV: |  Branch ([[@LINE-4]]:25): [True: 0, False: 0]
                           // BRCOV: |  Branch ([[@LINE-5]]:30): [True: 0, False: 0]
}


bool func2(int a, int b) {
    bool h = MACRO3 || COND4;  // BRCOV: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-32]]:15): [True: 1, False: [[#min(C,2)]]]
                               // BRCOV: |  |  |  |  |  |  |  |  |  |  |  Branch ([[@LINE-32]]:15): [True: 0, False: 1]
                               // BRCOV: |  |  |  |  |  |  |  Branch ([[@LINE-34]]:15): [True: 1, False: [[#min(C,2)]]]
                               // BRCOV: |  |  |  |  |  |  |  Branch ([[@LINE-34]]:15): [True: 0, False: 1]
                               // BRCOV: |  |  |  Branch ([[@LINE-33]]:15): [True: 1, False: [[#min(C,2)]]]
  return h;
}


int main(int argc, char *argv[])
{
  func(atoi(argv[1]), atoi(argv[2]));
  func2(atoi(argv[1]), atoi(argv[2]));
  (void)0;
  return 0;
}
