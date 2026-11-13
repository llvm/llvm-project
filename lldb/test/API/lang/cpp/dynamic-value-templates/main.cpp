
struct Interface {
  virtual ~Interface() = default;
  int intf = 321;
};

template <typename A> struct Template1 : public Interface {
  int templ = 1;
};

template <typename A, typename B> struct Template2 : public Interface {
  int templ = 2;
};

template <typename A, int B> struct TypeAndConstant : public Interface {
  int templ = 3;
};

template <typename A> struct Single {};
template <typename A, typename B> struct Double {};

int main() {
  Interface *t1_int = new Template1<int>;
  Interface *t1_single = new Template1<Single<int>>;
  Interface *t1_singleception = new Template1<Single<Single<int>>>;
  Interface *t1_double = new Template1<Double<int, float>>;

  Interface *t2_int_bool = new Template2<int, bool>;
  Interface *t2_single_double = new Template2<Single<int>, Double<int, int>>;
  Interface *t2_complex =
      new Template2<Single<int>, Double<Single<bool>, Single<char>>>;

  Interface *type_and_const = new TypeAndConstant<int, 42>;

  return 0; // break here
}
