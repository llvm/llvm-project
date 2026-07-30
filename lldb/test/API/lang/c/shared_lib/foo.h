struct foo;

struct sub_foo
{
  int sub_1;
  char *sub_2;
};

LLDB_TEST_API struct foo *GetMeAFoo();
LLDB_TEST_API struct sub_foo *GetMeASubFoo(struct foo *in_foo);
