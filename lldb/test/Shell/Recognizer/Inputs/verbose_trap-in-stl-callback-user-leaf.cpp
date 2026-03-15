void definitely_aborts() { __builtin_verbose_trap("User", "Invariant violated"); }

namespace std {
void aborts_soon() { definitely_aborts(); }
} // namespace std

void g() { std::aborts_soon(); }

namespace std {
namespace detail {
void eventually_aborts() { g(); }
} // namespace detail

inline namespace __1 {
void eventually_aborts() { detail::eventually_aborts(); }
} // namespace __1
} // namespace std

int main() {
  std::eventually_aborts();
  return 0;
}
