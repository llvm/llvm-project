// Test that we don't crash when trying to pretty-print structures that don't
// have the layout our data formatters expect.
namespace std {
template<typename T, typename Deleter = void>
class unique_ptr {};
}

int main() {
  std::unique_ptr<int> U;
  return 0; //% self.expect("frame variable U", substrs=["unique_ptr", "{}"])
}
