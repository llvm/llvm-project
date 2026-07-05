struct ContextClass {
  int member = 3;
  ContextClass *this_type = nullptr;
  ContextClass() { this_type = this; }

  int func() const {
    return member; // break in function in class.
  }

  template <class T> T templateFunc(T x) const {
    return member; // break in templated function in class.
  }
};

template <typename TC> struct TemplatedContextClass {
  int member = 4;
  TemplatedContextClass<TC> *this_type = nullptr;
  TemplatedContextClass() { this_type = this; }

  int func() const {
    return member; // break in function in templated class.
  }

  template <class T> T templateFunc(T x) const {
    return member; // break in templated function in templated class.
  }
};

int main() {
  ContextClass c;
  TemplatedContextClass<int> t;
  return c.func() + c.templateFunc(1) + t.func() + t.templateFunc(1);
}
