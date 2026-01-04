// Helper that the lldb command `statistics dump` works in split-dwarf mode.

struct Baz {
  int x;
  bool y;
};

void baz() {
  Baz b;
  b.x = 1;
  b.y = true;
}
