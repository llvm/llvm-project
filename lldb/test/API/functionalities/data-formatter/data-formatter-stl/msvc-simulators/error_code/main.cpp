// Layout approximation of MSVC STL std::error_code / error_condition.

namespace std {
struct error_code {
  int _Myval;
  const void *_Mycat;
};

struct error_condition {
  int _Myval;
  const void *_Mycat;
};
} // namespace std

int main() {
  std::error_code ec{2, nullptr};
  std::error_condition econd{7, nullptr};
  return 0; // break here
}
