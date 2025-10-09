



#include <stdio.h>
#include <stdlib.h>
// CHECK: | [[#min(C,4)]]|void func(
void func(int a, int b) {
  bool b0 = a <= b;
  bool b1 = a == b;
  bool b2 = a >= b;
  bool b3 = a < b;
  bool b4 = a > b;
  bool b5 = a != b;

  bool c = b0 &&           // BRCOV: Branch ([[@LINE]]:12): [True: [[#min(C,3)]], False: 1]
           b1 &&           // BRCOV: Branch ([[@LINE]]:12): [True: [[#min(C,2)]], False: 1]
           b2 &&           // BRCOV: Branch ([[@LINE]]:12): [True: [[#min(C,2)]], False: 0]
           b3 &&           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: [[#min(C,2)]]]
           b4 &&           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b5;             // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]

  bool d = b0 ||           // BRCOV: Branch ([[@LINE]]:12): [True: [[#min(C,3)]], False: 1]
           b1 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 1]
           b2 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 1, False: 0]
           b3 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b4 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b5;             // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]

  bool e = (b0  &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,3)]], False: 1]
            b5) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[#min(C,2)]]]
           (b1  &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,2)]], False: 1]
            b4) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[#min(C,2)]]]
           (b2  &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,3)]], False: 0]
            b3) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[#min(C,3)]]]
           (b3  &&         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[#min(C,3)]]]
            b2) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 0]
           (b4  &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[#min(C,2)]]]
            b1) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 1]
           (b5  &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[#min(C,2)]]]
            b0);           // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 1]

  bool f = (b0  ||         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,3)]], False: 1]
            b5) &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: 0]
           (b1  ||         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,2)]], False: [[#min(C,2)]]]
            b4) &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: 1]
           (b2  ||         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,3)]], False: 0]
            b3) &&         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 0]
           (b3  ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[#min(C,3)]]]
            b2) &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,3)]], False: 0]
           (b4  ||         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[#min(C,2)]]]
            b1) &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,2)]], False: 0]
           (b5  ||         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[#min(C,2)]]]
            b0);           // BRCOV: Branch ([[@LINE]]:13): [True: [[#min(C,2)]], False: 0]

  if (c)                   // BRCOV: Branch ([[@LINE]]:7): [True: 0, False: [[#min(C,4)]]]
    printf("case0\n");
  else
    printf("case1\n");

  if (d)                   // BRCOV: Branch ([[@LINE]]:7): [True: [[#min(C,4)]], False: 0]
    printf("case2\n");
  else
    printf("case3\n");

  if (e)                   // BRCOV: Branch ([[@LINE]]:7): [True: 1, False: [[#min(C,3)]]]
    printf("case4\n");
  else
    printf("case5\n");

  if (f)                   // BRCOV: Branch ([[@LINE]]:7): [True: [[#min(C,3)]], False: 1]
    printf("case6\n");
  else
    printf("case7\n");
}


int main(int argc, char *argv[])
{
  func(atoi(argv[1]), atoi(argv[2]));
  (void)0;
  return 0;
}
