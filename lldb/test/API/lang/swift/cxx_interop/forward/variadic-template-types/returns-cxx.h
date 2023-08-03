
template <class... Ts> struct Tuple {};

template <>
struct Tuple<> {
  void set() {}
};

template <class T, class... Ts>
struct Tuple<T, Ts...> : Tuple<Ts...> {
  Tuple(T t, Ts... ts) : Tuple<Ts...>(ts...), _t(t) {}

  void set(T t, Ts... ts) { _t = t; Tuple<Ts...>::set(ts...); }

  T first() { return _t; }
  Tuple<Ts...> rest() { return *this; }

  T _t;
};

struct CxxClass {
  long long a1 = 10;
  long long a2 = 20;
  long long a3 = 30;
};

struct OtherCxxClass {
  bool v = false;
};

typedef Tuple<CxxClass, OtherCxxClass> Pair;

inline Pair returnPair() { return Pair(CxxClass(), OtherCxxClass()); }

inline Tuple<OtherCxxClass, CxxClass> returnVariadic() { return Tuple<OtherCxxClass, CxxClass>(OtherCxxClass(), CxxClass()); }
