#include <stdio.h>
#include <stdlib.h>
struct mytype {
  int c;
  int d;
};

union myunion {
  int num;
  char *str;
};

typedef struct mytype MyType;

int main()
{
  struct mytype v;
  MyType *v_ptr = &v;

  union myunion u = {5};
  v.c = u.num;
  v.d = 10;
  return v.c + v.d;
}

