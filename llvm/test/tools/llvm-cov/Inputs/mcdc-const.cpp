#include <stdio.h>








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

    return 0;
}
