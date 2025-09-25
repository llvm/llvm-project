#define COMPRESSED_PAIR_REV 4
#include <libcxx-simulators-common/compressed_pair.h>

namespace std {
inline namespace __1 {
template <typename T> struct vector {
  T *__begin_;
  T *__end_;
  _LLDB_COMPRESSED_PAIR(T *, __cap_ = nullptr, void *, __alloc_);
};
} // namespace __1

inline namespace __2 {
template <typename T> struct vector {};
} // namespace __2

inline namespace __3 {
template <typename T> struct vector {
  T *__begin_;
  T *__end_;
  _LLDB_COMPRESSED_PAIR(short *, __cap_ = nullptr, void *, __alloc_);
};
} // namespace __3
} // namespace std

int main() {
  int arr[] = {1, 2, 3};
  std::__1::vector<int> v1{.__begin_ = arr, .__end_ = nullptr};
  std::__1::vector<int> v2{.__begin_ = nullptr, .__end_ = arr};
  std::__1::vector<int> v3{.__begin_ = &arr[2], .__end_ = arr};
  std::__2::vector<int> v4;

  char carr[] = {'a'};
  std::__3::vector<char> v5{.__begin_ = carr, .__end_ = carr + 1};

  return 0;
}
