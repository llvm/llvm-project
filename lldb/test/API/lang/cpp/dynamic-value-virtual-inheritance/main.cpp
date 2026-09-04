
namespace {
template <typename A, typename B> struct Pair {};
} // namespace

struct Interface {
  virtual int Foo() { return 0; }
};

struct Base : public Interface {
  bool is_base = true;
};

template <typename A, typename B> struct Templated : virtual public Base {
  bool is_templated = true;
};

template <typename A, int B> struct Complicated : virtual public Base {
  bool is_complicated = true;
};

template <typename A>
struct Evil : public Templated<int, double>,
              public Complicated<Pair<Interface, int>, 42> {
  bool is_evil = true;
};

int main() {
  auto lambda = [] {};

  Interface *base = new Base();
  Interface *templated = new Templated<int, double>();
  Interface *complicated = new Complicated<Pair<Interface, int>, 42>();
  Interface *evil = new Evil<decltype(lambda)>();
  return 0; // break here
}
