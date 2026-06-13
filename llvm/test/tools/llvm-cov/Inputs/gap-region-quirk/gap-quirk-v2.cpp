// A scoped block with a never-taken early return followed by at least one
// statement is needed to produce the gap region pattern. Without the extra
// statement after the if-block, clang emits a region entry on the line after
// the closing "}" (MinRegionCount > 0) and the bug doesn't trigger. With the
// extra statement, the closing "}" produces a gap region that wraps forward,
// and the next line has only a non-entry segment (MinRegionCount == 0).
bool process(bool err) {
  {
    if (err) {
      return false;
    }
    int fd = 1;
    (void)fd;
  }
  int result = 42;
  (void)result;
  return true;
}

int main() {
  process(false);
  return 0;
}
