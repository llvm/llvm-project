//===--- span- The span class -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ACXXEL_SPAN_H
#define ACXXEL_SPAN_H

#include <array>
#include <cstddef>
#include <exception>
#include <iterator>
#include <type_traits>

namespace acxxel {

/// Value used to indicate slicing to the end of the span.
static constexpr std::ptrdiff_t dynamic_extent = -1; // NOLINT

class SpanBase {};

/// Implementation of the proposed C++17 std::span class.
///
/// Based on the paper:
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0122r1.pdf
template <typename ElementType> class Span : public SpanBase {
public:
  /// \name constants and types
  /// \{

  using element_type = ElementType;
  using index_type = std::ptrdiff_t;
  using pointer = element_type *;
  using reference = element_type &;
  using iterator = element_type *;
  using const_iterator = const element_type *;
  using value_type = typename std::remove_const<element_type>::type;

  /// \}

  /// \name constructors, copy, assignment, and destructor.
  /// \{

  /// Constructs an empty span with null pointer data.
  Span() : Data(nullptr), Size(0) {}

  /// Constructs an empty span with null pointer data.
  // Intentionally implicit.
  Span(std::nullptr_t) : Data(nullptr), Size(0) {}

  /// Constructs a span from a pointer and element count.
  Span(pointer Ptr, index_type Count) : Data(Ptr), Size(Count) {
    if (Count < 0 || (!Ptr && Count))
      std::terminate();
  }

  /// Constructs a span from a pointer to the fist element in the range and a
  /// pointer to one past the last element in the range.
  Span(pointer FirstElem, pointer LastElem)
      : Data(FirstElem), Size(std::distance(FirstElem, LastElem)) {
    if (Size < 0)
      std::terminate();
  }

  /// Constructs a span from an array.
  // Intentionally implicit.
  template <typename T, size_t N> Span(T (&Arr)[N]) : Data(Arr), Size(N) {}

  /// Constructs a span from a std::array.
  // Intentionally implicit.
  template <size_t N>
  Span(const std::array<typename std::remove_const<element_type>::type, N> &Arr)
      : Data(Arr.data()), Size(N) {}

  /// Constructs a span from a container such as a std::vector.
  // TODO(jhen): Put in a check to make sure this constructor does not
  // participate in overload resolution unless Container meets the following
  // requirements:
  //  * Container is a contiguous container and a sequence container.
  // Intentionally implicit.
  template <typename Container>
  Span(Container &Cont,
       typename std::enable_if<
           std::is_same<
               typename std::remove_const<typename Container::value_type>::type,
               typename std::remove_const<element_type>::type>::value &&
           !std::is_array<Container>::value &&
           !std::is_base_of<SpanBase, Container>::value &&
           std::is_convertible<decltype(&Cont[0]), pointer>::value>::type * =
           nullptr)
      : Data(Cont.data()), Size(Cont.size()) {}

  /// Avoids creating spans from expiring temporary objects.
  // TODO(jhen): Put in a check to make sure this constructor does not
  // participate in overload resolution unless Container meets the following
  // requirements:
  //  * Container is a contiguous container and a sequence container.
  template <typename Container>
  Span(Container &&Cont,
       typename std::enable_if<
           std::is_same<
               typename std::remove_const<typename Container::value_type>::type,
               typename std::remove_const<element_type>::type>::value &&
           !std::is_array<Container>::value &&
           !std::is_base_of<SpanBase, Container>::value &&
           std::is_convertible<decltype(&Cont[0]), pointer>::value>::type * =
           nullptr) = delete;

  Span(const Span &) noexcept = default;
  Span(Span &&) noexcept;

  /// Constructs a span from copying a span of another type that can be
  /// implicitly converted to the type stored by the constructed span.
  // Intentionally implicit.
  template <typename OtherElementType>
  Span(const Span<OtherElementType> &Other)
      : Data(Other.Data), Size(Other.Size) {}

  /// Constructs a span from moving a span of another type that can be
  /// implicitly converted to the type stored by the constructed span.
  // Intentionally implicit.
  template <typename OtherElementType>
  Span(Span<OtherElementType> &&Other) : Data(Other.Data), Size(Other.Size) {}

  ~Span() = default;

  Span &operator=(const Span &) noexcept = default;
  Span &operator=(Span &&) noexcept;

  /// \}

  /// \name subviews
  /// \{

  /// Creates a span out of the first Count elements of this span.
  Span<element_type> first(index_type Count) const {
    bool Valid = Count >= 0 && Count <= size();
    if (!Valid)
      std::terminate();
    return Span<element_type>(data(), Count);
  }

  /// Creates a span out of the last Count elements of this span.
  Span<element_type> last(index_type Count) const {
    bool Valid = Count >= 0 && Count <= size();
    if (!Valid)
      std::terminate();
    return Span<element_type>(Count == 0 ? data() : data() + (size() - Count),
                              Count);
  }

  /// Creates a span out of the Count elements of this span beginning at Offset.
  ///
  /// If no arguments is provided for Count, the new span will extend to the end
  /// of the current span.
  Span<element_type> subspan(index_type Offset,
                             index_type Count = dynamic_extent) const {
    bool Valid =
        (Offset == 0 || (Offset > 0 && Offset <= size())) &&
        (Count == dynamic_extent || (Count >= 0 && Offset + Count <= size()));
    if (!Valid)
      std::terminate();
    return Span<element_type>(
        data() + Offset, Count == dynamic_extent ? size() - Offset : Count);
  }

  /// \}

  /// \name observers
  /// \{

  index_type length() const { return Size; }
  index_type size() const { return Size; }
  bool empty() const { return size() == 0; }

  /// \}

  /// \name element access
  /// \{

  reference operator[](index_type Idx) const {
    bool Valid = Idx >= 0 && Idx < size();
    if (!Valid)
      std::terminate();
    return Data[Idx];
  }

  reference operator()(index_type Idx) const { return operator[](Idx); }

  pointer data() const noexcept { return Data; }

  /// \}

  /// \name iterator support
  /// \{

  iterator begin() const noexcept { return Data; }
  iterator end() const noexcept { return Data + Size; }
  const_iterator cbegin() const noexcept { return Data; }
  const_iterator cend() const noexcept { return Data + Size; }

  /// \}

private:
  template <typename OtherElementType> friend class Span;

  pointer Data;
  index_type Size;
};

template <typename ElementType>
Span<ElementType>::Span(Span &&) noexcept = default;
template <typename ElementType>
Span<ElementType> &Span<ElementType>::operator=(Span &&) noexcept = default;

} // namespace acxxel

#endif // ACXXEL_SPAN_H
