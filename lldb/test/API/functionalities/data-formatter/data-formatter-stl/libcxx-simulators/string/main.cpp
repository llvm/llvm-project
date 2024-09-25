#include <libcxx-simulators-common/compressed_pair.h>

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
#ifdef SUBCLASS_PADDING
    struct : __padding<value_type> {
      unsigned char __size_;
    };
#else // !SUBCLASS_PADDING

    unsigned char __padding[sizeof(value_type) - 1];
#ifdef BITMASKS
    unsigned char __size_;
#else // !BITMASKS
    unsigned char __size_ : 7;
    unsigned char __is_long_ : 1;
#endif // BITMASKS
#endif // SUBCLASS_PADDING
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

  __long &getLongRep() {
#if COMPRESSED_PAIR_REV == 0
    return __r_.first().__l;
#elif COMPRESSED_PAIR_REV <= 2
    return __rep_.__l;
#endif
  }

  __short &getShortRep() {
#if COMPRESSED_PAIR_REV == 0
    return __r_.first().__s;
#elif COMPRESSED_PAIR_REV <= 2
    return __rep_.__s;
#endif
  }

#if COMPRESSED_PAIR_REV == 0
  std::__lldb::__compressed_pair<__rep, allocator_type> __r_;
#elif COMPRESSED_PAIR_REV <= 2
  _LLDB_COMPRESSED_PAIR(__rep, __rep_, allocator_type, __alloc_);
#endif

public:
  template <size_t __N>
  basic_string(unsigned char __size, const value_type (&__data)[__N]) {
    static_assert(__N < __min_cap, "");
#ifdef BITMASKS
    getShortRep().__size_ = __size << __short_shift;
#else
    getShortRep().__size_ = __size;
    getShortRep().__is_long_ = false;
#endif
    for (size_t __i = 0; __i < __N; ++__i)
      getShortRep().__data_[__i] = __data[__i];
  }
  basic_string(size_t __cap, size_type __size, pointer __data) {
#ifdef BITMASKS
    getLongRep().__cap_ = __cap | __long_mask;
#else
    getLongRep().__cap_ = __cap / __endian_factor;
    getLongRep().__is_long_ = true;
#endif
    getLongRep().__size_ = __size;
    getLongRep().__data_ = __data;
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
