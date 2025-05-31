#include <stdio.h>








void test(bool a, bool b, bool c, bool d) {

  if ((a && b) || (c && d))
    printf("test1 decision true\n");

  if (b && c) if (a && d)
    printf("test2 decision true\n");

  if ((c && d) &&
      (a && b))
    printf("test3 decision true\n");
}

int main()
{
    test(false,false,false,false);
    test(true,false,true,false);
    test(true,false,true,true);
    test(true,true,false,false);

    test(true,false,false,false);
    test(true,true,true,true);
    test(false,true,true,false);

    (void)0;
    return 0;
}
