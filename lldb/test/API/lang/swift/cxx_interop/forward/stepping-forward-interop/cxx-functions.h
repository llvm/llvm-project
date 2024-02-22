void cxxFunction();

struct CxxClass {
  void cxxMethod();
};

struct ClassWithConstructor {
  int a;
  bool b;
  double c;

  ClassWithConstructor(int a, bool b, double c);

};

struct ClassWithExtension {
  int val = 42;

  int definedInExtension();
};

struct ClassWithCallOperator {
  int operator()(); 
};


