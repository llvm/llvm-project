#ifndef A_H
#define A_H

#include <cstdio>
#include <memory>

class A {
public:
  A(int value) : m_a_value(value) {}
  A(int value, A *client_A) : m_a_value(value), m_client_A(client_A) {}

  virtual ~A() {}

  virtual void doSomething(A &anotherA);

  int Value() { return m_a_value; }

private:
  int m_a_value;
  std::auto_ptr<A> m_client_A;
};

A *make_anonymous_B();

#endif
