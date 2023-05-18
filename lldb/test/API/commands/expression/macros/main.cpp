#include "macro1.h"

#define MACRO_1 100
#define MACRO_2 200

int
main ()
{
  int a = ONE + TWO; // Break here

  #undef MACRO_2
  #undef FOUR

  return Simple().Method();
}
