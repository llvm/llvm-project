#include "a.h"

namespace {
class B : public A {
public:
  B() : A(42) {}

private:
  int m_anon_b_value = 47;
};
} // namespace

A *make_anonymous_B() { return new B(); }
