namespace ns {
  typedef int id;
};

int bar() {
  int id = 12;
  int Class = 15;
  return id + Class;
}

int main()
{
  ns::id foo = 0;
  return foo + bar(); // Set breakpoint here.
}
