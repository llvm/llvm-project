#define COMPRESSED_PAIR_REV 4
#include <libcxx-simulators-common/compressed_pair.h>
#include <stddef.h>

namespace std {
inline namespace __ValidLegacyVector {
template <typename T> struct vector {
  T *__begin_;
  T *__end_;
};
} // namespace __ValidLegacyVector

inline namespace __LegacyVectorMissingBegin {
template <typename T> struct vector {
  T *__end_;
};
} // namespace __LegacyVectorMissingBegin

inline namespace __LegacyVectorNonPointerBegin {
template <typename T> struct vector {
  int __begin_;
  T *__end_;
};
} // namespace __LegacyVectorNonPointerBegin

inline namespace __LegacyMissingEnd {
template <typename T> struct vector {
  T *__begin_;
};
} // namespace __LegacyMissingEnd

inline namespace __LegacyVectorNonPointerEnd {
template <typename T> struct vector {
  T *__begin_;
  size_t __end_;
};
} // namespace __LegacyVectorNonPointerEnd

inline namespace __LegacyVectorSizeBased {
template <typename T> struct vector {
  T *__begin_;
  size_t __end_;
};
} // namespace __LegacyVectorSizeBased

inline namespace __ValidPointerLayout {
template <typename T> struct __vector_layout {
  T *__begin_;
  T *__end_;
};

template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __ValidPointerLayout

inline namespace __PointerLayoutNonPointerBegin {
template <typename T> struct __vector_layout {
  size_t __begin_;
  T *__end_;
};

template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __PointerLayoutNonPointerBegin

inline namespace __PointerLayoutNonPointerEnd {
template <typename T> struct __vector_layout {
  T *__begin_;
  size_t __end_;
};

template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __PointerLayoutNonPointerEnd

inline namespace __LayoutStructMissingBegin {
template <typename T> struct __vector_layout {
  // LLDB short-circuits when it can't find `__begin_`, so other members aren't
  // required for this type.
};

template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __LayoutStructMissingBegin

inline namespace __LayoutStructMissingSecondMember {
template <typename T> struct __vector_layout {
  T *__begin_;
};

template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __LayoutStructMissingSecondMember

inline namespace __ValidSizeLayout {
template <typename T> struct __vector_layout {
  T *__begin_;
  size_t __size_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __ValidSizeLayout

inline namespace __SizeLayoutMissingBegin {
template <typename T> struct __vector_layout {
  size_t __size_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __SizeLayoutMissingBegin

inline namespace __SizeLayoutNonPointerBegin {
template <typename T> struct __vector_layout {
  size_t __begin_;
  size_t __size_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __SizeLayoutNonPointerBegin

inline namespace __SizeLayoutNonIntegerSize {
template <typename T> struct __vector_layout {
  T *__begin_;
  T *__size_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __SizeLayoutNonIntegerSize
} // namespace std

int main() {
  int arr[] = {1, 2, 3};
  std::__ValidLegacyVector::vector<int> v1{.__begin_ = arr, .__end_ = nullptr};
  std::__ValidLegacyVector::vector<int> v2{.__begin_ = nullptr, .__end_ = arr};
  std::__ValidLegacyVector::vector<int> v3{.__begin_ = &arr[2], .__end_ = arr};
  std::__LegacyVectorMissingBegin::vector<int> v4{.__end_ = arr};
  std::__LegacyMissingEnd::vector<int> v5{.__begin_ = arr};
  std::__LegacyVectorNonPointerBegin::vector<int> v6{.__begin_ = 0,
                                                     .__end_ = arr};
  std::__LegacyVectorNonPointerEnd::vector<int> v7{.__begin_ = arr,
                                                   .__end_ = 0};

  std::__LayoutStructMissingBegin::vector<int> v8{.__layout_ = {}};
  std::__LayoutStructMissingSecondMember::vector<int> v9{
      .__layout_ = {.__begin_ = arr}};

  std::__ValidPointerLayout::vector<int> v10{
      .__layout_ = {.__begin_ = arr, .__end_ = nullptr}};
  std::__ValidPointerLayout::vector<int> v11{
      .__layout_ = {.__begin_ = nullptr, .__end_ = arr}};
  std::__ValidPointerLayout::vector<int> v12{
      .__layout_ = {.__begin_ = &arr[2], .__end_ = arr}};

  std::__PointerLayoutNonPointerBegin::vector<int> v13{
      .__layout_ = {.__begin_ = 0, .__end_ = arr}};
  std::__PointerLayoutNonPointerEnd::vector<int> v14{
      .__layout_ = {.__begin_ = arr, .__end_ = 0}};

  std::__ValidSizeLayout::vector<int> v15{
      .__layout_ = {.__begin_ = arr, .__size_ = 1}};

  std::__SizeLayoutMissingBegin::vector<int> v16{.__layout_ = {.__size_ = 1}};
  std::__SizeLayoutNonPointerBegin::vector<int> v17{
      .__layout_ = {.__begin_ = 0, .__size_ = 0}};
  std::__SizeLayoutNonIntegerSize::vector<int> v18{
      .__layout_ = {.__begin_ = arr, .__size_ = 0}};

  char carr[] = {'a'};
  std::__ValidLegacyVector::vector<short> v19{
      .__begin_ = reinterpret_cast<short *>(carr),
      .__end_ = reinterpret_cast<short *>(carr + 1)};
  std::__ValidPointerLayout::vector<short> v20{
      .__layout_ = {.__begin_ = reinterpret_cast<short *>(carr),
                    .__end_ = reinterpret_cast<short *>(carr + 1)}};
  std::__ValidSizeLayout::vector<short> v21{
      .__layout_ = {.__begin_ = reinterpret_cast<short *>(carr), .__size_ = 1}};

  struct ZeroSizeStruct {
    int x[0];
  };
  static_assert(sizeof(ZeroSizeStruct) == 0);

  std::__ValidLegacyVector::vector<ZeroSizeStruct> v23{.__begin_ = nullptr,
                                                       .__end_ = nullptr};
  std::__ValidPointerLayout::vector<ZeroSizeStruct> v24{
      .__layout_ = {.__begin_ = nullptr, .__end_ = nullptr}};
  std::__ValidSizeLayout::vector<ZeroSizeStruct> v25{
      .__layout_ = {.__begin_ = nullptr, .__size_ = 0}};
  return 0;
}
