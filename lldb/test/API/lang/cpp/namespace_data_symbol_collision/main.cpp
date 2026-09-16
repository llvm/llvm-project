// `colliding_ns` is a real (inline) namespace in this translation unit.
// Two unrelated internal data symbols of the same name live in
// colliding_a.o / colliding_b.o (built without debug info).
inline namespace colliding_ns {
int do_thing(int t) { return t + 1; }
} // namespace colliding_ns

extern "C" const int *colliding_a_anchor(void);
extern "C" const int *colliding_b_anchor(void);

int main() {
  // Force the linker to keep both internal `colliding_ns` data symbols.
  (void)colliding_a_anchor();
  (void)colliding_b_anchor();

  int r = do_thing(5);
  return r; // Break here
}
