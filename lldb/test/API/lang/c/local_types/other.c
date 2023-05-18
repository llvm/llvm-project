struct Foo {
  char *ptr1;
  char *ptr2;
};

int
callme(int input)
{
  struct Foo myFoo = { "string one", "Set a breakpoint in other"};
  return myFoo.ptr1[0] + myFoo.ptr2[0] + input;
}
