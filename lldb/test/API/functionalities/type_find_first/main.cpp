namespace Integer {
struct Point {
  int x, y;
};
} // namespace Integer

namespace Float {
struct Point {
  float x, y;
};
} // namespace Float

int main(int argc, char const *argv[]) {
  Integer::Point ip = {2, 3};
  Float::Point fp = {2.0, 3.0};
  return 0;
}
