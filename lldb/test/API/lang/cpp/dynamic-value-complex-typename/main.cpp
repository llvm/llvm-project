
struct Interface {
  virtual int Foo() { return 0; }
};

struct Simple : public Interface {
  bool is_simple = true;
};

template <typename A, typename B> struct Templated : public Interface {
  bool is_templated = true;
};

template <typename A, int B> struct Complicated : public Interface {
  bool is_complicated = true;
};

template <typename A> struct Evil : public Interface {
  bool is_evil = true;
};

namespace {
template <typename A, typename B> struct Pair {};
} // namespace

int main() {
  auto lambda = [] {};

  Interface *simple = new Simple();
  Interface *templated = new Templated<int, double>();
  Interface *complicated = new Complicated<Pair<Interface, int>, 42>();
  Interface *evil = new Evil<decltype(lambda)>();
  return 0; // break here
}
