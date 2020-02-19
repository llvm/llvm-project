struct OuterX {
  template<typename T>
  struct Inner {
    int oX_inner = 42;
  };
};

struct OuterY {
  template<typename T>
  struct Inner {
    typename OuterX::Inner<T> oY_inner;
  };
};

struct WrapperB;

WrapperB* foo();
