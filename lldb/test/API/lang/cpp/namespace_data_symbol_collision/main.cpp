#include <cstdio>

// `colliding_ns` is a real (inline) namespace in this translation unit.
// Two unrelated internal data symbols of the same name live in
// colliding_a.o / colliding_b.o (built without debug info).
inline namespace colliding_ns {
template <typename T> int do_thing(T const &t) { return t.mem; }
} // namespace colliding_ns

struct S {
  int mem;
};

extern "C" const int *colliding_a_anchor(void);
extern "C" const int *colliding_b_anchor(void);

int main() {
  // Force the linker to keep both internal `colliding_ns` data symbols.
  (void)colliding_a_anchor();
  (void)colliding_b_anchor();

  int r = do_thing(S{.mem = 42});
  std::puts("Break here");
  return r;
}
