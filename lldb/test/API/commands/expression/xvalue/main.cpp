struct Tmp
{
  int data = 1234;
};

Tmp foo() { return Tmp(); }

int main(int argc, char const *argv[])
{
  int something = foo().data;
  return 0; // Break here
}
