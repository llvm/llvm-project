template<typename T>
int foo(T t1) {
        return int(t1);
}

// Some cases to cover ADL, we have two cases:
//
// - f which will have a overload in the global namespace if unqualified lookup
// find f(int) and f(T) is found via ADL.
//
// - g which does not have an overload in the global namespace.
namespace A {
struct C {};

template <typename T> int f(T) { return 4; }

template <typename T> int g(T) { return 4; }
} // namespace A

// Meant to overload A::f(T) which may be found via ADL
int f(int) { return 1; }

// Regular overloaded functions case h(T) and h(double).
template <class T> int h(T x) { return x; }
int h(double d) { return 5; }

template <class... Us> int var(Us... pargs) { return 10; }

// Having the templated overloaded operators in a namespace effects the
// mangled name generated in the IR e.g. _ZltRK1BS1_ Vs _ZN1AltERKNS_1BES2_
// One will be in the symbol table but the other won't. This results in a
// different code path that will result in CPlusPlusNameParser being used.
// This allows us to cover that code as well.
namespace A {
template <typename T> bool operator<(const T &, const T &) { return true; }

template <typename T> bool operator>(const T &, const T &) { return true; }

template <typename T> bool operator<<(const T &, const T &) { return true; }

template <typename T> bool operator>>(const T &, const T &) { return true; }

template <typename T> bool operator==(const T &, const T &) { return true; }

struct B {};
} // namespace A

struct D {};

// Make sure we cover more straight forward cases as well.
bool operator<(const D &, const D &) { return true; }
bool operator>(const D &, const D &) { return true; }
bool operator>>(const D &, const D &) { return true; }
bool operator<<(const D &, const D &) { return true; }
bool operator==(const D &, const D &) { return true; }

int main() {
  A::B b1;
  A::B b2;
  D d1;
  D d2;

  bool result_b = b1 < b2 && b1 << b2 && b1 == b2 && b1 > b2 && b1 >> b2;
  bool result_c = d1 < d2 && d1 << d2 && d1 == d2 && d1 > d2 && d1 >> d2;

  return foo(42) + result_b + result_c + f(A::C{}) + g(A::C{}) + h(10) + h(1.) +
         var(1) + var(1, 2); // break here
}
