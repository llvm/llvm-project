static int g_next_value = 12345;

struct VBase {
  VBase() : m_value(g_next_value++) {}
  virtual ~VBase() {}
  int m_value;
};

struct Derived1 : public virtual VBase {
};

struct Derived2 : public virtual VBase {
};

struct Joiner1 : public Derived1, public Derived2 {
  long x = 1;
};

struct Joiner2 : public Derived2 {
  long y = 2;
};

int main(int argc, const char *argv[]) {
  Joiner1 j1;
  Joiner2 j2;
  Derived2 *d = &j1;
  d = &j2;  // breakpoint 1
  return 0; // breakpoint 2
}
