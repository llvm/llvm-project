// The stack is shallower at the first stop than at the second, so the outermost
// frames have to be renumbered in between.

void shallow(void) {
  int shallow_local = 1;
  (void)shallow_local; // Shallow breakpoint.
}

void deep(void) {
  int deep_local = 2;
  (void)deep_local; // Deep breakpoint.
}

void middle(void) { deep(); }

int main(void) {
  shallow();
  middle();
  return 0;
}
