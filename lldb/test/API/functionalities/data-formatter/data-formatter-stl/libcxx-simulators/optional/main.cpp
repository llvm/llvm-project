#include <type_traits>
#include <utility>

#if REVISION == 0
// Pre-a3942b3 layout.
#define NO_REMOVE_CV
#endif
// REVISION == 1: current layout

namespace std {
namespace __lldb {

struct in_place_t {
  explicit in_place_t() = default;
};
constexpr in_place_t in_place{};

template <class _Tp, bool = is_trivially_destructible<_Tp>::value>
struct __optional_destruct_base {
  typedef _Tp value_type;
  union {
    char __null_state_;
#ifdef NO_REMOVE_CV
    value_type __val_;
#else // !NO_REMOVE_CV
    remove_cv_t<value_type> __val_;
#endif
  };
  bool __engaged_;

  template <class... _Args>
  constexpr explicit __optional_destruct_base(in_place_t, _Args &&...__args)
      : __val_(std::forward<_Args>(__args)...), __engaged_(true) {}
};

template <class _Tp, bool = is_reference<_Tp>::value>
struct __optional_storage_base : __optional_destruct_base<_Tp> {
  using __base = __optional_destruct_base<_Tp>;
  using value_type = _Tp;
  using __base::__base;
};

template <class _Tp, bool = is_trivially_copy_constructible<_Tp>::value>
struct __optional_copy_base : __optional_storage_base<_Tp> {
  using __optional_storage_base<_Tp>::__optional_storage_base;
};

template <class _Tp, bool = is_trivially_move_constructible<_Tp>::value>
struct __optional_move_base : __optional_copy_base<_Tp> {
  using __optional_copy_base<_Tp>::__optional_copy_base;
};

template <class _Tp, bool = is_trivially_destructible<_Tp>::value &&
                            is_trivially_copy_constructible<_Tp>::value &&
                            is_trivially_copy_assignable<_Tp>::value>
struct __optional_copy_assign_base : __optional_move_base<_Tp> {
  using __optional_move_base<_Tp>::__optional_move_base;
};

template <class _Tp, bool = is_trivially_destructible<_Tp>::value &&
                            is_trivially_move_constructible<_Tp>::value &&
                            is_trivially_move_assignable<_Tp>::value>
struct __optional_move_assign_base : __optional_copy_assign_base<_Tp> {
  using __optional_copy_assign_base<_Tp>::__optional_copy_assign_base;
};

template <bool _CanCopy, bool _CanMove> struct __sfinae_ctor_base {};

template <class _Tp>
using __optional_sfinae_ctor_base_t =
    __sfinae_ctor_base<is_copy_constructible<_Tp>::value,
                       is_move_constructible<_Tp>::value>;

template <bool _CanCopy, bool _CanMove> struct __sfinae_assign_base {};

template <class _Tp>
using __optional_sfinae_assign_base_t = __sfinae_assign_base<
    (is_copy_constructible<_Tp>::value && is_copy_assignable<_Tp>::value),
    (is_move_constructible<_Tp>::value && is_move_assignable<_Tp>::value)>;

template <class _Tp>
class optional : private __optional_move_assign_base<_Tp>,
                 private __optional_sfinae_ctor_base_t<_Tp>,
                 private __optional_sfinae_assign_base_t<_Tp> {
  using __base = __optional_move_assign_base<_Tp>;

public:
  using value_type = _Tp;

public:
  template <class _Up = value_type>
  constexpr explicit optional(_Up &&__v)
      : __base(in_place, std::forward<_Up>(__v)) {}
};

} // namespace __lldb
} // namespace std

int main() {
  std::__lldb::optional<char const *> maybe_string{"Hello"};
  std::__lldb::optional<int> maybe_int{42};
  return 0; // Break here
}
