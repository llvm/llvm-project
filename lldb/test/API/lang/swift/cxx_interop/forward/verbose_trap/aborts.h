#include <iterator>

namespace std {
void function_that_aborts() { __builtin_verbose_trap("Error", "from C++"); }

struct ConstIterator {
private:
  int value;

public:
  // Make sure this auto-conforms to UnsafeCxxInputIterator

  using iterator_category = std::input_iterator_tag;
  using value_type = int;
  using pointer = int *;
  using reference = const int &;
  using difference_type = int;

  ConstIterator(int value) : value(value) {}

  void operator*() const { std::function_that_aborts(); }

  ConstIterator &operator++() { return *this; }
  ConstIterator operator++(int) { return ConstIterator(value); }

  bool operator==(const ConstIterator &other) const { return false; }
  bool operator!=(const ConstIterator &other) const { return true; }
};

} // namespace std
