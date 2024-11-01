struct Base {
  int m_base_val = 42;
};

struct Foo : public Base {
  int m_derived_val = 137;
};

int main() {
  Foo foo;
  return 0;
}
