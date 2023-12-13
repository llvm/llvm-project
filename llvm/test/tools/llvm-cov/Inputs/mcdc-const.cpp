
#include <stdio.h>

extern "C" {
extern void __llvm_profile_write_file(void);
}

extern int foo();

void test(bool a, bool b, bool c, bool d) {

  if ((a && 1) || (0 && d) || 0)
    printf("test1 decision true\n");
}

int main()
{
    test(true,false,true,false);
    test(true,false,true,true);
    test(true,true,false,false);
    test(false,true,true,false);

    test(true,false,false,false);

    __llvm_profile_write_file();
    return 0;
}
