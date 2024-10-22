#include <libcxx-simulators-common/compressed_pair.h>

#include <stdio.h>

namespace std {
namespace __lldb {
template <class _Tp> struct default_delete {
  default_delete() noexcept = default;

  void operator()(_Tp *__ptr) const noexcept { delete __ptr; }
};

template <class _Tp, class _Dp = default_delete<_Tp>> class unique_ptr {
public:
  typedef _Tp element_type;
  typedef _Dp deleter_type;
  typedef _Tp *pointer;

#if COMPRESSED_PAIR_REV == 0
  std::__lldb::__compressed_pair<pointer, deleter_type> __ptr_;
  explicit unique_ptr(pointer __p) noexcept
      : __ptr_(__p, std::__lldb::__value_init_tag()) {}
#elif COMPRESSED_PAIR_REV == 1 || COMPRESSED_PAIR_REV == 2
  _LLDB_COMPRESSED_PAIR(pointer, __ptr_, deleter_type, __deleter_);
  explicit unique_ptr(pointer __p) noexcept : __ptr_(__p), __deleter_() {}
#endif
};
} // namespace __lldb
} // namespace std

struct StatefulDeleter {
  StatefulDeleter() noexcept = default;

  void operator()(int *__ptr) const noexcept { delete __ptr; }

  int m_state = 50;
};

int main() {
  std::__lldb::unique_ptr<int> var_up(new int(5));
  std::__lldb::unique_ptr<int, StatefulDeleter> var_with_deleter_up(new int(5));
  __builtin_printf("Break here\n");
  return 0;
}
