#ifndef NUMBER_H
#define NUMBER_H

#include <cstddef>

template <std::size_t Size> class Numbers {
public:
  using ValueType = int;
  using PointerType = ValueType *;
  using ReferenceType = ValueType &;

  class Iterator {
    // all the operators needs to be inlined so that there is no call
    // instruction.
  public:
    __attribute__((always_inline)) explicit Iterator(PointerType ptr)
        : m_ptr(ptr) {}

    __attribute__((always_inline)) Iterator &operator++() noexcept {
      ++m_ptr;
      return *this;
    };

    __attribute__((always_inline)) Iterator operator++(int) noexcept {
      Iterator iter = *this;
      m_ptr++;
      return iter;
    }

    __attribute__((always_inline)) ReferenceType operator*() const noexcept {
      return *m_ptr;
    }
    __attribute__((always_inline)) bool
    operator==(const Iterator &iter) noexcept {
      return m_ptr == iter.m_ptr;
    }
    __attribute__((always_inline)) bool
    operator!=(const Iterator &iter) noexcept {
      return !(*this == iter);
    }

  private:
    PointerType m_ptr;
  };

  PointerType data() { return static_cast<PointerType>(m_buffer); }

  Iterator begin() { return Iterator(data()); }
  Iterator cbegin() { return Iterator(data()); }
  Iterator end() { return Iterator(data() + Size); }
  Iterator cend() { return Iterator(data() + Size); }

private:
  int m_buffer[Size]{};
};

#endif // NUMBER_H