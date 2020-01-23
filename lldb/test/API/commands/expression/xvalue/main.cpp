struct Tmp
{
  int data = 1234;
};

Tmp foo() { return Tmp(); }

int main() { int argc = 0; char **argv = (char **)0;

  int something = foo().data;
  return 0; // Break here
}
