#include <climits>
#include <memory>
#include <type_traits>

#if REVISION == 0
// Pre-c3d0205ee771 layout.
#define SUBCLASS_PADDING
#endif
#if REVISION <= 1
// Pre-D123580 layout.
#define BITMASKS
#endif
#if REVISION <= 2
// Pre-D125496 layout.
#define SHORT_UNION
#endif
#if REVISION == 3
// Pre-D128285 layout.
#define PACKED_ANON_STRUCT
#endif
// REVISION == 4: current layout

#ifdef PACKED_ANON_STRUCT
#define BEGIN_PACKED_ANON_STRUCT struct __attribute__((packed)) {
#define END_PACKED_ANON_STRUCT };
#else
#define BEGIN_PACKED_ANON_STRUCT
#define END_PACKED_ANON_STRUCT
#endif


namespace std {
namespace __lldb {

template <class _Tp, int _Idx,
          bool _CanBeEmptyBase =
              std::is_empty<_Tp>::value && !std::is_final<_Tp>::value>
struct __compressed_pair_elem {
  explicit __compressed_pair_elem(_Tp __t) : __value_(__t) {}

  _Tp &__get() { return __value_; }

private:
  _Tp __value_;
};

template <class _Tp, int _Idx>
struct __compressed_pair_elem<_Tp, _Idx, true> : private _Tp {
  explicit __compressed_pair_elem(_Tp __t) : _Tp(__t) {}

  _Tp &__get() { return *this; }
};

template <class _T1, class _T2>
class __compressed_pair : private __compressed_pair_elem<_T1, 0>,
                          private __compressed_pair_elem<_T2, 1> {
public:
  using _Base1 = __compressed_pair_elem<_T1, 0>;
  using _Base2 = __compressed_pair_elem<_T2, 1>;

  explicit __compressed_pair(_T1 __t1, _T2 __t2) : _Base1(__t1), _Base2(__t2) {}

  _T1 &first() { return static_cast<_Base1 &>(*this).__get(); }
};

#if defined(ALTERNATE_LAYOUT) && defined(SUBCLASS_PADDING)
template <class _CharT, size_t = sizeof(_CharT)> struct __padding {
  unsigned char __xx[sizeof(_CharT) - 1];
};

template <class _CharT> struct __padding<_CharT, 1> {};
#endif

template <class _CharT, class _Traits, class _Allocator> class basic_string {
public:
  typedef _CharT value_type;
  typedef _Allocator allocator_type;
  typedef allocator_traits<allocator_type> __alloc_traits;
  typedef typename __alloc_traits::size_type size_type;
  typedef typename __alloc_traits::pointer pointer;

#ifdef ALTERNATE_LAYOUT

  struct __long {
    pointer __data_;
    size_type __size_;
#ifdef BITMASKS
    size_type __cap_;
#else
    size_type __cap_ : sizeof(size_type) * CHAR_BIT - 1;
    size_type __is_long_ : 1;
#endif
  };

  enum {
    __min_cap = (sizeof(__long) - 1) / sizeof(value_type) > 2
                    ? (sizeof(__long) - 1) / sizeof(value_type)
                    : 2
  };

  struct __short {
    value_type __data_[__min_cap];
#ifdef BITMASKS
#ifdef SUBCLASS_PADDING
    struct : __padding<value_type> {
      unsigned char __size_;
    };
#else
    unsigned char __padding[sizeof(value_type) - 1];
    unsigned char __size_;
#endif
#else // !BITMASKS
    unsigned char __size_ : 7;
    unsigned char __is_long_ : 1;
#endif
  };

#ifdef BITMASKS
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  static const size_type __short_shift = 1;
  static const size_type __long_mask = 0x1ul;
#else
  static const size_type __short_shift = 0;
  static const size_type __long_mask = ~(size_type(~0) >> 1);
#endif
#else // !BITMASKS
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  static const size_type __endian_factor = 2;
#else
  static const size_type __endian_factor = 1;
#endif
#endif // BITMASKS

#else // !ALTERNATE_LAYOUT

  struct __long {
#ifdef BITMASKS
    size_type __cap_;
#else
    BEGIN_PACKED_ANON_STRUCT
    size_type __is_long_ : 1;
    size_type __cap_ : sizeof(size_type) * CHAR_BIT - 1;
    END_PACKED_ANON_STRUCT
#endif
    size_type __size_;
    pointer __data_;
  };

  enum {
    __min_cap = (sizeof(__long) - 1) / sizeof(value_type) > 2
                    ? (sizeof(__long) - 1) / sizeof(value_type)
                    : 2
  };

  struct __short {
#ifdef SHORT_UNION
    union {
#ifdef BITMASKS
      unsigned char __size_;
#else
      struct {
        unsigned char __is_long_ : 1;
        unsigned char __size_ : 7;
      };
#endif
      value_type __lx;
    };
#else
    BEGIN_PACKED_ANON_STRUCT
    unsigned char __is_long_ : 1;
    unsigned char __size_ : 7;
    END_PACKED_ANON_STRUCT
    char __padding_[sizeof(value_type) - 1];
#endif
    value_type __data_[__min_cap];
  };

#ifdef BITMASKS
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  static const size_type __short_shift = 0;
  static const size_type __long_mask = ~(size_type(~0) >> 1);
#else
  static const size_type __short_shift = 1;
  static const size_type __long_mask = 0x1ul;
#endif
#else // !BITMASKS
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  static const size_type __endian_factor = 1;
#else
  static const size_type __endian_factor = 2;
#endif
#endif

#endif // ALTERNATE_LAYOUT

  union __ulx {
    __long __lx;
    __short __lxx;
  };

  enum { __n_words = sizeof(__ulx) / sizeof(size_type) };

  struct __raw {
    size_type __words[__n_words];
  };

  struct __rep {
    union {
      __long __l;
      __short __s;
      __raw __r;
    };
  };

  __compressed_pair<__rep, allocator_type> __r_;

public:
  template <size_t __N>
  basic_string(unsigned char __size, const value_type (&__data)[__N])
      : __r_({}, {}) {
    static_assert(__N < __min_cap, "");
#ifdef BITMASKS
    __r_.first().__s.__size_ = __size << __short_shift;
#else
    __r_.first().__s.__size_ = __size;
    __r_.first().__s.__is_long_ = false;
#endif
    for (size_t __i = 0; __i < __N; ++__i)
      __r_.first().__s.__data_[__i] = __data[__i];
  }
  basic_string(size_t __cap, size_type __size, pointer __data) : __r_({}, {}) {
#ifdef BITMASKS
    __r_.first().__l.__cap_ = __cap | __long_mask;
#else
    __r_.first().__l.__cap_ = __cap / __endian_factor;
    __r_.first().__l.__is_long_ = true;
#endif
    __r_.first().__l.__size_ = __size;
    __r_.first().__l.__data_ = __data;
  }
};

using string = basic_string<char, std::char_traits<char>, std::allocator<char>>;

} // namespace __lldb
} // namespace std

int main() {
  char longdata[] = "I am a very long string";
  std::__lldb::string longstring(sizeof(longdata), sizeof(longdata) - 1,
                                 longdata);
  std::__lldb::string shortstring(5, "short");
  return 0; // Break here
}
