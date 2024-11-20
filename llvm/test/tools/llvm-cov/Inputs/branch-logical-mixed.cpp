



#include <stdio.h>
#include <stdlib.h>
// CHECK: |{{ +}}[[C4:4|1]]|void func(
void func(int a, int b) {
  bool b0 = a <= b;
  bool b1 = a == b;
  bool b2 = a >= b;
  bool b3 = a < b;
  bool b4 = a > b;
  bool b5 = a != b;

  bool c = b0 &&           // BRCOV: Branch ([[@LINE]]:12): [True: [[C3:3|1]], False: 1]
           b1 &&           // BRCOV: Branch ([[@LINE]]:12): [True: [[C2:2|1]], False: 1]
           b2 &&           // BRCOV: Branch ([[@LINE]]:12): [True: [[C2]], False: 0]
           b3 &&           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: [[C2]]]
           b4 &&           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b5;             // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]

  bool d = b0 ||           // BRCOV: Branch ([[@LINE]]:12): [True: [[C3]], False: 1]
           b1 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 1]
           b2 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 1, False: 0]
           b3 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b4 ||           // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b5;             // BRCOV: Branch ([[@LINE]]:12): [True: 0, False: 0]

  bool e = (b0  &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[C3]], False: 1]
            b5) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[C2]]]
           (b1  &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[C2]], False: 1]
            b4) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[C2]]]
           (b2  &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[C3]], False: 0]
            b3) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[C3]]]
           (b3  &&         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[C3]]]
            b2) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 0]
           (b4  &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[C2]]]
            b1) ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 1]
           (b5  &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[C2]]]
            b0);           // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 1]

  bool f = (b0  ||         // BRCOV: Branch ([[@LINE]]:13): [True: [[C3]], False: 1]
            b5) &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: 0]
           (b1  ||         // BRCOV: Branch ([[@LINE]]:13): [True: [[C2]], False: [[C2]]]
            b4) &&         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: 1]
           (b2  ||         // BRCOV: Branch ([[@LINE]]:13): [True: [[C3]], False: 0]
            b3) &&         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: 0]
           (b3  ||         // BRCOV: Branch ([[@LINE]]:13): [True: 0, False: [[C3]]]
            b2) &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[C3]], False: 0]
           (b4  ||         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[C2]]]
            b1) &&         // BRCOV: Branch ([[@LINE]]:13): [True: [[C2]], False: 0]
           (b5  ||         // BRCOV: Branch ([[@LINE]]:13): [True: 1, False: [[C2]]]
            b0);           // BRCOV: Branch ([[@LINE]]:13): [True: [[C2]], False: 0]

  if (c)                   // CHECK: Branch ([[@LINE]]:7): [True: 0, False: [[C4]]]
    printf("case0\n");
  else
    printf("case1\n");

  if (d)                   // CHECK: Branch ([[@LINE]]:7): [True: [[C4]], False: 0]
    printf("case2\n");
  else
    printf("case3\n");

  if (e)                   // CHECK: Branch ([[@LINE]]:7): [True: 1, False: [[C3:3|1]]]
    printf("case4\n");
  else
    printf("case5\n");

  if (f)                   // CHECK: Branch ([[@LINE]]:7): [True: [[C3]], False: 1]
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
