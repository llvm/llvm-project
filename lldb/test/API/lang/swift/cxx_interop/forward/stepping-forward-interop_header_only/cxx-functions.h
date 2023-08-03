inline void cxxFunction() {
    int a = 42;
}

struct CxxClass {
  void cxxMethod() {
    int a = 42;
  }
};

struct ClassWithConstructor {
  int a;
  bool b;
  double c;

  ClassWithConstructor(int a, bool b, double c): a(a), b(b), c(c) {
  }
};

struct ClassWithExtension {
  int val = 42;

  int definedInExtension() {
    return val;
  }
};

struct ClassWithCallOperator {
  int operator()() {
    return 42;
  }
};


