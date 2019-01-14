//==------------ multi_ptr.hpp - SYCL multi_ptr class ----------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>
#include <cassert>
#include <cstddef>

namespace cl {
namespace sycl {
template <typename ElementType, access::address_space Space> class multi_ptr {
public:
  using element_type = ElementType;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer and reference types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = typename detail::PtrValueType<ElementType, Space>::type *;
  using const_pointer_t =
      typename detail::PtrValueType<ElementType, Space>::type const *;
  using reference_t = typename detail::PtrValueType<ElementType, Space>::type &;
  using const_reference_t =
      typename detail::PtrValueType<ElementType, Space>::type &;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &rhs) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr(ElementType *pointer)
      : m_Pointer(reinterpret_cast<pointer_t>(pointer)) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#endif
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr &operator=(ElementType *pointer) {
    m_Pointer = reinterpret_cast<pointer_t>(pointer);
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#endif
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }
  ElementType &operator*() const {
    return *(reinterpret_cast<ElementType *>(m_Pointer));
  }
  ElementType *operator->() const {
    return reinterpret_cast<ElementType *>(m_Pointer);
  }
  ElementType &operator[](difference_type index) {
    return *(reinterpret_cast<ElementType *>(m_Pointer + index));
  }
  ElementType operator[](difference_type index) const {
    return *(reinterpret_cast<ElementType *>(m_Pointer + index));
  }

  // Only if Space == global_space
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder,
            access::address_space _Space = Space,
            typename = typename std::enable_if<
                _Space == Space &&
                Space == access::address_space::global_space>::type>
  multi_ptr(accessor<ElementType, dimensions, Mode,
                     access::target::global_buffer, isPlaceholder>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space
  template <
      int dimensions, access::mode Mode, access::placeholder isPlaceholder,
      access::address_space _Space = Space,
      typename = typename std::enable_if<
          _Space == Space && Space == access::address_space::local_space>::type>
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::local,
                     isPlaceholder>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <int dimensions, access::mode Mode,
            access::placeholder isPlaceholder,
            access::address_space _Space = Space,
            typename = typename std::enable_if<
                _Space == Space &&
                Space == access::address_space::constant_space>::type>
  multi_ptr(accessor<ElementType, dimensions, Mode,
                     access::target::constant_buffer, isPlaceholder>
                Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }

  // Implicit conversion to the underlying pointer type
  operator ElementType *() const {
    return reinterpret_cast<ElementType *>(m_Pointer);
  }

  // Explicit conversion to a multi_ptr<void>
  explicit operator multi_ptr<void, Space>() const;

  // Arithmetic operators
  multi_ptr &operator++() {
    m_Pointer += (difference_type)1;
    return *this;
  }
  multi_ptr operator++(int) {
    multi_ptr result(*this);
    ++(*this);
    return result;
  }
  multi_ptr &operator--() {
    m_Pointer -= (difference_type)1;
    return *this;
  }
  multi_ptr operator--(int) {
    multi_ptr result(*this);
    --(*this);
    return result;
  }
  multi_ptr &operator+=(difference_type r) {
    m_Pointer += r;
    return *this;
  }
  multi_ptr &operator-=(difference_type r) {
    m_Pointer -= r;
    return *this;
  }
  multi_ptr operator+(difference_type r) const {
    return multi_ptr(m_Pointer + r);
  }
  multi_ptr operator-(difference_type r) const {
    return multi_ptr(m_Pointer - r);
  }

  void prefetch(size_t numElements) const;

private:
  pointer_t m_Pointer;
};

// Specialization of multi_ptr for void
template <access::address_space Space> class multi_ptr<void, Space> {
public:
  using element_type = void;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = typename detail::PtrValueType<void, Space>::type *;
  using const_pointer_t =
      typename detail::PtrValueType<void, Space>::type const *;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() : m_Pointer(nullptr) {}
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t pointer) : m_Pointer(pointer) {}
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr(void *pointer) : m_Pointer(reinterpret_cast<pointer_t>(pointer)) {
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#endif
  multi_ptr(std::nullptr_t) : m_Pointer(nullptr) {}
  ~multi_ptr() = default;

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t pointer) {
    m_Pointer = pointer;
    return *this;
  }
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr &operator=(void *pointer) {
    m_Pointer = reinterpret_cast<pointer_t>(pointer);
    // TODO An implementation should reject an argument if the deduced
    // address space is not compatible with Space.
  }
#endif
  multi_ptr &operator=(std::nullptr_t) {
    m_Pointer = nullptr;
    return *this;
  }

  // Only if Space == global_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space _Space = Space,
            typename = typename std::enable_if<
                _Space == Space &&
                Space == access::address_space::global_space>::type>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::global_buffer>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == local_space
  template <
      typename ElementType, int dimensions, access::mode Mode,
      access::address_space _Space = Space,
      typename = typename std::enable_if<
          _Space == Space && Space == access::address_space::local_space>::type>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::local> Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Only if Space == constant_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space _Space = Space,
            typename = typename std::enable_if<
                _Space == Space &&
                Space == access::address_space::constant_space>::type>
  multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::constant_buffer>
          Accessor)
      : multi_ptr(Accessor.get_pointer()) {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_Pointer; }

  // Implicit conversion to the underlying pointer type
  operator void *() const;

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, Space>() const;

private:
  pointer_t m_Pointer;
};

template <typename ElementType, access::address_space Space>
multi_ptr<ElementType, Space>
make_ptr(typename multi_ptr<ElementType, Space>::pointer_t pointer) {
  return multi_ptr<ElementType, Space>(pointer);
}

#ifdef __SYCL_DEVICE_ONLY__
// An implementation should reject an argument if the deduced address space
// is not compatible with Space.
// This is guaranteed by the c'tor.
template <typename ElementType, access::address_space Space>
multi_ptr<ElementType, Space> make_ptr(ElementType *pointer) {
  return multi_ptr<ElementType, Space>(pointer);
}
#endif

template <typename ElementType, access::address_space Space>
bool operator==(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator!=(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator<(const multi_ptr<ElementType, Space> &lhs,
               const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator>(const multi_ptr<ElementType, Space> &lhs,
               const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator<=(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator>=(const multi_ptr<ElementType, Space> &lhs,
                const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator!=(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t rhs);

template <typename ElementType, access::address_space Space>
bool operator!=(std::nullptr_t lhs, const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator==(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t rhs);

template <typename ElementType, access::address_space Space>
bool operator==(std::nullptr_t lhs, const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator>(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t rhs);

template <typename ElementType, access::address_space Space>
bool operator>(std::nullptr_t lhs, const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator<(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t rhs);

template <typename ElementType, access::address_space Space>
bool operator<(std::nullptr_t lhs, const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator>=(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t rhs);

template <typename ElementType, access::address_space Space>
bool operator>=(std::nullptr_t lhs, const multi_ptr<ElementType, Space> &rhs);

template <typename ElementType, access::address_space Space>
bool operator<=(const multi_ptr<ElementType, Space> &lhs, std::nullptr_t rhs);

template <typename ElementType, access::address_space Space>
bool operator<=(std::nullptr_t lhs, const multi_ptr<ElementType, Space> &rhs);
} // namespace sycl
} // namespace cl
