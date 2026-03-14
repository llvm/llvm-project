class A {
public:
  static int a;
  int b;
};

class B {
public:
  static int a;
  int b;
};

struct C {
  static int a;
};

int A::a = 1111;
int B::a = 2222;
int C::a = 3333;
int a = 4444;

int main() // break here
{
  return 0;
}
