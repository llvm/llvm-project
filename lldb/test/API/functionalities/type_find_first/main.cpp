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

namespace OtherCompilationUnit {
void Function();
} // namespace OtherCompilationUnit

int main(int argc, char const *argv[]) {
  Integer::Point ip = {2, 3};
  Float::Point fp = {2.0, 3.0};
  OtherCompilationUnit::Function();
  return 0;
}
