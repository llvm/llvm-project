#include <stddef.h>

namespace std {
template <class T, class E> class expected {
public:
  expected(const T &value) : _Value(value), _Has_value(true) {}
  expected(const E &error, bool) : _Unexpected(error), _Has_value(false) {}
  union {
    T _Value;
    E _Unexpected;
  };
  bool _Has_value;
};

template <class E> class expected<void, E> {
public:
  expected() : _Has_value(true) {}
  expected(const E &error, bool) : _Unexpected(error), _Has_value(false) {}
  union {
    E _Unexpected;
  };
  bool _Has_value;
};
} // namespace std

int main() {
  std::expected<int, const char *> ok(7);
  std::expected<int, const char *> err("boom", true);
  std::expected<void, int> void_ok;
  std::expected<void, int> void_err(11, true);
  return 0; // break here
}
