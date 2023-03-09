
template<class T>
struct Wrapper {
  T t;
  Wrapper(T t) : t(t) {}
};

struct CxxClass {
  long long a1 = 10;
  long long a2 = 20;
  long long a3 = 30;
};

inline Wrapper<CxxClass> returnWrapper() { return Wrapper<CxxClass>(CxxClass()); }
