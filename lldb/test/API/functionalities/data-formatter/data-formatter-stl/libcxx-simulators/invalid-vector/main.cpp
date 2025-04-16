#define COMPRESSED_PAIR_REV 2
#include <libcxx-simulators-common/compressed_pair.h>

namespace std {
namespace __1 {
template <typename T> struct vector {
  T *__begin_;
  T *__end_;
  _LLDB_COMPRESSED_PAIR(T *, __cap_ = nullptr, void *, __alloc_);
};
} // namespace __1

namespace __2 {
template <typename T> struct vector {};
} // namespace __2
} // namespace std

int main() {
  int arr[] = {1, 2, 3};
  std::__1::vector<int> v1{.__begin_ = arr, .__end_ = nullptr};
  std::__1::vector<int> v2{.__begin_ = nullptr, .__end_ = arr};
  std::__1::vector<int> v3{.__begin_ = &arr[3], .__end_ = arr};
  std::__2::vector<int> v4;

  return 0;
}
