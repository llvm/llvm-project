#include <stdio.h>

typedef struct Struct
{
  int one;
  int two;
} Struct;

int
main()
{
  Struct myStruct = {10, 20};
  printf ("Break here: %d\n.", myStruct.one);
  return 0;
}
