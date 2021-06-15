int other();

namespace {
template <typename T1> struct Temp { int x; };
// This emits the 'Temp' template in this TU.
Temp<float> Template1;
} // namespace

int main() {
  return Template1.x + other(); // break here
}
