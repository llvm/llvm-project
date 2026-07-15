// The main executable is a macCatalyst (arm64-apple-ios-macabi) binary that
// calls into a plain macOS Swift framework via a C entry point.
extern void entry(void);

int main() {
  entry();
  return 0;
}
