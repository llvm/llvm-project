#include "foo.h"
#include <stdlib.h>

struct foo
{
  struct sub_foo sub_element;
  int    other_element;
};

struct foo *
GetMeAFoo()
{
  struct foo *ret_val = (struct foo *) malloc (sizeof (struct foo));
  ret_val->other_element = 3;
  return ret_val;
}

struct sub_foo *
GetMeASubFoo (struct foo *in_foo)
{
  return &(in_foo->sub_element);
}
