#include "a.h"

void A::doSomething(A &anotherA) {
  printf("In A %p doing something with %d.\n", this, m_a_value);
  int tmp_value = anotherA.Value();
  printf("Also have another A at %p: %d.\n", &anotherA, tmp_value); // Break here in doSomething.
}

class Extra
{
public:
  Extra (int in_one, int in_two) : m_extra_one(in_one), m_extra_two(in_two) {}

private:
  int m_extra_one;
  int m_extra_two;
};

class B : public Extra, public virtual A
{
public:
  B (int b_value, int a_value) : Extra(b_value, a_value), A(a_value), m_b_value(b_value) {}
  B (int b_value, int a_value, A *client_A) : Extra(b_value, a_value), A(a_value, client_A), m_b_value(b_value) {}

  virtual ~B () {}

private:
  int m_b_value;
};

static A* my_global_A_ptr;

int
main (int argc, char **argv)
{
  my_global_A_ptr = new B (100, 200);
  B myB (10, 20, my_global_A_ptr);
  B *second_fake_A_ptr = new B (150, 250);
  B otherB (300, 400, second_fake_A_ptr);

  myB.doSomething(otherB); // Break here and get real addresses of myB and otherB.

  A reallyA (500);
  myB.doSomething (reallyA);  // Break here and get real address of reallyA.

  myB.doSomething(*make_anonymous_B());

  return 0;
}
