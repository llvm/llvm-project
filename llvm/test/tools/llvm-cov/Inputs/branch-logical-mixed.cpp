



#include <stdio.h>
#include <stdlib.h>

void func(int a, int b) {
  bool b0 = a <= b;
  bool b1 = a == b;
  bool b2 = a >= b;
  bool b3 = a < b;
  bool b4 = a > b;
  bool b5 = a != b;

  bool c = b0 &&           // CHECK: Branch ([[@LINE]]:12): [True: 3, False: 1]
           b1 &&           // CHECK: Branch ([[@LINE]]:12): [True: 2, False: 1]
           b2 &&           // CHECK: Branch ([[@LINE]]:12): [True: 2, False: 0]
           b3 &&           // CHECK: Branch ([[@LINE]]:12): [True: 0, False: 2]
           b4 &&           // CHECK: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b5;             // CHECK: Branch ([[@LINE]]:12): [True: 0, False: 0]

  bool d = b0 ||           // CHECK: Branch ([[@LINE]]:12): [True: 3, False: 1]
           b1 ||           // CHECK: Branch ([[@LINE]]:12): [True: 0, False: 1]
           b2 ||           // CHECK: Branch ([[@LINE]]:12): [True: 1, False: 0]
           b3 ||           // CHECK: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b4 ||           // CHECK: Branch ([[@LINE]]:12): [True: 0, False: 0]
           b5;             // CHECK: Branch ([[@LINE]]:12): [True: 0, False: 0]

  bool e = (b0  &&         // CHECK: Branch ([[@LINE]]:13): [True: 3, False: 1]
            b5) ||         // CHECK: Branch ([[@LINE]]:13): [True: 1, False: 2]
           (b1  &&         // CHECK: Branch ([[@LINE]]:13): [True: 2, False: 1]
            b4) ||         // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 2]
           (b2  &&         // CHECK: Branch ([[@LINE]]:13): [True: 3, False: 0]
            b3) ||         // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 3]
           (b3  &&         // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 3]
            b2) ||         // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 0]
           (b4  &&         // CHECK: Branch ([[@LINE]]:13): [True: 1, False: 2]
            b1) ||         // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 1]
           (b5  &&         // CHECK: Branch ([[@LINE]]:13): [True: 1, False: 2]
            b0);           // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 1]

  bool f = (b0  ||         // CHECK: Branch ([[@LINE]]:13): [True: 3, False: 1]
            b5) &&         // CHECK: Branch ([[@LINE]]:13): [True: 1, False: 0]
           (b1  ||         // CHECK: Branch ([[@LINE]]:13): [True: 2, False: 2]
            b4) &&         // CHECK: Branch ([[@LINE]]:13): [True: 1, False: 1]
           (b2  ||         // CHECK: Branch ([[@LINE]]:13): [True: 3, False: 0]
            b3) &&         // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 0]
           (b3  ||         // CHECK: Branch ([[@LINE]]:13): [True: 0, False: 3]
            b2) &&         // CHECK: Branch ([[@LINE]]:13): [True: 3, False: 0]
           (b4  ||         // CHECK: Branch ([[@LINE]]:13): [True: 1, False: 2]
            b1) &&         // CHECK: Branch ([[@LINE]]:13): [True: 2, False: 0]
           (b5  ||         // CHECK: Branch ([[@LINE]]:13): [True: 1, False: 2]
            b0);           // CHECK: Branch ([[@LINE]]:13): [True: 2, False: 0]

  if (c)                   // CHECK: Branch ([[@LINE]]:7): [True: 0, False: 4]
    printf("case0\n");
  else
    printf("case1\n");

  if (d)                   // CHECK: Branch ([[@LINE]]:7): [True: 4, False: 0]
    printf("case2\n");
  else
    printf("case3\n");

  if (e)                   // CHECK: Branch ([[@LINE]]:7): [True: 1, False: 3]
    printf("case4\n");
  else
    printf("case5\n");

  if (f)                   // CHECK: Branch ([[@LINE]]:7): [True: 3, False: 1]
    printf("case6\n");
  else
    printf("case7\n");
}

extern "C" { extern void __llvm_profile_write_file(void); }
int main(int argc, char *argv[])
{
  func(atoi(argv[1]), atoi(argv[2]));
  __llvm_profile_write_file();
  return 0;
}
