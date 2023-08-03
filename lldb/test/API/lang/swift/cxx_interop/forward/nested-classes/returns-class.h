
struct SuperClass {
  long long a = 10;
};

struct CxxClass {
  struct NestedClass {
    long long b = 20;
  };

  struct NestedSubclass: SuperClass {
    long long c = 30;
  };

};

