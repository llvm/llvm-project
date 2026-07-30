#define CLASS(NAME)                             \
  class NAME {                                  \
    public:                                     \
    class Inner {                               \
      int j = #NAME[0];                         \
    };                                          \
    Inner *i = nullptr;                         \
  };                                            \
                                                \
  static NAME::Inner inner;                     \
  static NAME obj;                              \
  NAME::Inner &getInner##NAME() { return inner; }
