namespace N0 {
namespace N1 {

namespace {
enum Enum { Enum_0 = 1, Enum_1 = 2, Enum_2 = 4, Enum_3 = 8 };
}

Enum Global = Enum_3;

struct Base {
  Enum m_e = Enum_1;
};

class Class : public Base {
public:
  Class(Enum e) : m_ce(e) {}

  static int StaticFunc(const Class &c) {
    return c.PrivateFunc(c.m_inner) + Global + ClassStatic;
  }

  const Enum m_ce;

  static int ClassStatic;

private:
  struct Inner {
    char x;
    short y;
    int z;
  };

  int PrivateFunc(const Inner &i) const { return i.z; }

  Inner m_inner{};
};
int Class::ClassStatic = 7;

template<typename T>
struct Template {
  template<Enum E>
  void TemplateFunc() {
    T::StaticFunc(T(E));
  }
};

void foo() { Template<Class>().TemplateFunc<Enum_0>(); }

} // namespace N1
} // namespace N0

int main() {
  N0::N1::foo();
  return 0;
}
