struct A {
public:
  static int getStaticValue() { return 5; }
  int getMemberValue() { return a; }
  int a;
};

int main()
{
  A a;
  a.a = 3;
  return A::getStaticValue() + a.getMemberValue(); // Break here
}
