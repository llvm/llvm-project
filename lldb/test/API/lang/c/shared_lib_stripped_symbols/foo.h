struct foo;

struct sub_foo
{
  int sub_1;
  char *sub_2;
};

struct foo *GetMeAFoo();
struct sub_foo *GetMeASubFoo (struct foo *in_foo);


