#include <stdio.h>

struct SubStruct
{
  int a;
  int b;
};

struct MyStruct
{
  int first;
  struct SubStruct second;
};

int
main()
{
  struct MyStruct my_struct = {10, {20, 30}};
  struct MyStruct *my_pointer = &my_struct;
  printf ("Stop here to evaluate expressions: %d %d %p\n", my_pointer->first, my_pointer->second.a, my_pointer);
  return 0;
}



