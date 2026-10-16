//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef REVERSE_MODE_AUTODIFF_MEMORY_MANAGEMENT_HPP
#define REVERSE_MODE_AUTODIFF_MEMORY_MANAGEMENT_HPP

#include <algorithm>
#include <array>
#include <boost/math/tools/assert.hpp>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>
namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {
namespace detail {
template<typename allocator_type, size_t buffer_size>
class flat_linear_allocator_iterator
{
  /**
   * @brief enables iterating over linear allocator with
   * c++ iterators
   */
public:
  using raw_allocator_type = std::remove_const_t<allocator_type>;
  using value_type = typename allocator_type::value_type;
  using pointer = typename allocator_type::value_type*;
  using const_ptr_type = const value_type*;
  using reference = typename allocator_type::value_type&;
  using const_reference_type = const value_type&;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = ptrdiff_t;

private:
  const allocator_type* storage_ = nullptr;
  size_t index_ = 0;
  size_t begin_ = 0;
  size_t end_ = 0;

public:
  flat_linear_allocator_iterator() = default;

  explicit flat_linear_allocator_iterator(allocator_type* storage, size_t index)
    : storage_(storage)
    , index_(index)
    , begin_(0)
    , end_(storage->size())
  {
  }

  explicit flat_linear_allocator_iterator(allocator_type* storage,
                                          size_t index,
                                          size_t begin,
                                          size_t end)
    : storage_(storage)
    , index_(index)
    , begin_(begin)
    , end_(end)
  {
  }

  explicit flat_linear_allocator_iterator(const allocator_type* storage,
                                          size_t index)
    : storage_(storage)
    , index_(index)
    , begin_(0)
    , end_(storage->size())
  {
  }

  explicit flat_linear_allocator_iterator(const allocator_type* storage,
                                          size_t index,
                                          size_t begin,
                                          size_t end)
    : storage_(storage)
    , index_(index)
    , begin_(begin)
    , end_(end)
  {
  }
  reference operator*()
  {
    BOOST_MATH_ASSERT(index_ >= begin_ && index_ < end_);
    return (*storage_->data_[index_ / buffer_size])[index_ % buffer_size];
  }

  const_reference_type operator*() const
  {
    BOOST_MATH_ASSERT(index_ >= begin_ && index_ < end_);
    return (*storage_->data_[index_ / buffer_size])[index_ % buffer_size];
  }

  pointer operator->()
  {
    BOOST_MATH_ASSERT(index_ >= begin_ && index_ < end_);
    return &operator*();
  }

  const_ptr_type operator->() const
  {
    BOOST_MATH_ASSERT(index_ >= begin_ && index_ < end_);
    return &operator*();
  }
  flat_linear_allocator_iterator& operator++()
  {
    ++index_;
    return *this;
  }

  flat_linear_allocator_iterator operator++(int)
  {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  flat_linear_allocator_iterator& operator--()
  {
    --index_;
    return *this;
  }

  flat_linear_allocator_iterator operator--(int)
  {
    auto tmp = *this;
    --(*this);
    return tmp;
  }

  bool operator==(const flat_linear_allocator_iterator& other) const
  {
    return index_ == other.index_ && storage_ == other.storage_;
  }

  bool operator!=(const flat_linear_allocator_iterator& other) const
  {
    return !(*this == other);
  }

  flat_linear_allocator_iterator operator+(difference_type n) const
  {
    return flat_linear_allocator_iterator(
      storage_, index_ + static_cast<size_t>(n), begin_, end_);
  }

  flat_linear_allocator_iterator& operator+=(difference_type n)
  {
    index_ += n;
    return *this;
  }

  flat_linear_allocator_iterator operator-(difference_type n) const
  {
    return flat_linear_allocator_iterator(storage_, index_ - n, begin_, end_);
  }
  flat_linear_allocator_iterator& operator-=(difference_type n)
  {
    index_ -= n;
    return *this;
  }

  difference_type operator-(const flat_linear_allocator_iterator& other) const
  {
    return static_cast<difference_type>(index_) -
           static_cast<difference_type>(other.index_);
  }

  reference operator[](difference_type n) { return *(*this + n); }

  const_reference_type operator[](difference_type n) const
  {
    return *(*this + n);
  }

  bool operator<(const flat_linear_allocator_iterator& other) const
  {
    return index_ < other.index_;
  }

  bool operator>(const flat_linear_allocator_iterator& other) const
  {
    return index_ > other.index_;
  }

  bool operator<=(const flat_linear_allocator_iterator& other) const
  {
    return index_ <= other.index_;
  }

  bool operator>=(const flat_linear_allocator_iterator& other) const
  {
    return index_ >= other.index_;
  }

  bool operator!() const noexcept { return storage_ == nullptr; }
};
/* memory management helps for tape */
template<typename RealType, size_t buffer_size>
class flat_linear_allocator
{
  /** @brief basically a vector<array<T*, size>>
   * intended to work like a vector that allocates memory in chunks
   * and doesn't invalidate references
   * */
public:
  // store vector of unique pointers to arrays
  // to avoid vector reference invalidation
  using buffer_type = std::array<RealType, buffer_size>;
  using buffer_ptr = std::unique_ptr<std::array<RealType, buffer_size>>;

private:
  std::vector<buffer_ptr> data_;
  size_t total_size_ = 0;
  std::vector<size_t> checkpoints_; //{0};

public:
  friend class flat_linear_allocator_iterator<
    flat_linear_allocator<RealType, buffer_size>,
    buffer_size>;
  friend class flat_linear_allocator_iterator<
    const flat_linear_allocator<RealType, buffer_size>,
    buffer_size>;
  using value_type = RealType;
  using iterator =
    flat_linear_allocator_iterator<flat_linear_allocator<RealType, buffer_size>,
                                   buffer_size>;
  using const_iterator = flat_linear_allocator_iterator<
    const flat_linear_allocator<RealType, buffer_size>,
    buffer_size>;

  size_t buffer_id() const noexcept { return total_size_ / buffer_size; }
  size_t item_id() const noexcept { return total_size_ % buffer_size; }

private:
  void allocate_buffer()
  {
    data_.emplace_back(std::make_unique<buffer_type>());
  }

public:
  flat_linear_allocator() { allocate_buffer(); }
  flat_linear_allocator(const flat_linear_allocator&) = delete;
  flat_linear_allocator& operator=(const flat_linear_allocator&) = delete;
  flat_linear_allocator(flat_linear_allocator&&) = delete;
  flat_linear_allocator& operator=(flat_linear_allocator&&) = delete;
  ~flat_linear_allocator()
  {
    destroy_all();
    data_.clear();
  }

  void destroy_all()
  {
    for (size_t i = 0; i < total_size_; ++i) {
      size_t bid = i / buffer_size;
      size_t iid = i % buffer_size;
      (*data_[bid])[iid].~RealType();
    }
  }
  /** @brief
   * helper functions to clear tape and create block in tape
   */
  void clear()
  {
    data_.clear();
    total_size_ = 0;
    checkpoints_.clear();
    allocate_buffer();
  }

  // doesn't delete anything, only sets the current index to zero
  void reset() { total_size_ = 0; }
  void rewind() { total_size_ = 0; };

  // adds current index as a checkpoint to be able to walk back to
  void add_checkpoint()
  {
    checkpoints_.push_back(total_size_);
    /*
     * if (total_size_ > 0) {
      checkpoints_.push_back(total_size_ - 1);
    } else {
      checkpoints_.push_back(0);
    }*/
  };

  /** @brief clears all checkpoints
   * */
  void reset_checkpoints() { checkpoints_.clear(); }

  void rewind_to_last_checkpoint() { total_size_ = checkpoints_.back(); }
  void rewind_to_checkpoint_at(size_t index)
  {
    total_size_ = checkpoints_[index];
  }

  void fill(const RealType& val)
  {
    for (size_t i = 0; i < total_size_; ++i) {
      size_t bid = i / buffer_size;
      size_t iid = i % buffer_size;
      (*data_[bid])[iid] = val;
    }
  }

  /** @brief emplaces back object at the end of the
   * data structure, calls default constructor */
  iterator emplace_back()
  {
    if (item_id() == 0 && total_size_ != 0) {
      allocate_buffer();
    }
    size_t bid = buffer_id();
    size_t iid = item_id();

    RealType* ptr = &(*data_[bid])[iid];
    new (ptr) RealType();
    ++total_size_;
    return iterator(this, total_size_ - 1);
  };

  /** @brief, emplaces back object at end of data structure,
   * passes arguments to constructor */
  template<typename... Args>
  iterator emplace_back(Args&&... args)
  {
    if (item_id() == 0 && total_size_ != 0) {
      allocate_buffer();
    }
    BOOST_MATH_ASSERT(buffer_id() < data_.size());
    BOOST_MATH_ASSERT(item_id() < buffer_size);
    RealType* ptr = &(*data_[buffer_id()])[item_id()];
    new (ptr) RealType(std::forward<Args>(args)...);
    ++total_size_;
    return iterator(this, total_size_ - 1);
  }
  /** @brief default constructs n objects at end of
   * data structure, n known at compile time */
  template<size_t n>
  iterator emplace_back_n()
  {
    size_t bid = buffer_id();
    size_t iid = item_id();
    if (iid + n < buffer_size) {
      RealType* ptr = &(*data_[bid])[iid];
      for (size_t i = 0; i < n; ++i) {
        new (ptr + i) RealType();
      }
      total_size_ += n;
      return iterator(this, total_size_ - n, total_size_ - n, total_size_);
    } else {
      size_t allocs_in_curr_buffer = buffer_size - iid;
      size_t allocs_in_next_buffer = n - (buffer_size - iid);
      RealType* ptr = &(*data_[bid])[iid];
      for (size_t i = 0; i < allocs_in_curr_buffer; ++i) {
        new (ptr + i) RealType();
      }
      allocate_buffer();
      bid = data_.size() - 1;
      iid = 0;
      total_size_ += n;

      RealType* ptr2 = &(*data_[bid])[iid];
      for (size_t i = 0; i < allocs_in_next_buffer; i++) {
        new (ptr2 + i) RealType();
      }
      return iterator(this, total_size_ - n, total_size_ - n, total_size_);
    }
  }
  /** @brief default constructs n objects at end of
   * data structure, n known at run time
   */
  iterator emplace_back_n(size_t n)
  {
    size_t bid = buffer_id();
    size_t iid = item_id();
    if (iid + n < buffer_size) {
      RealType* ptr = &(*data_[bid])[iid];
      for (size_t i = 0; i < n; ++i) {
        new (ptr + i) RealType();
      }
      total_size_ += n;
      return iterator(this, total_size_ - n, total_size_ - n, total_size_);
    } else {
      size_t allocs_in_curr_buffer = buffer_size - iid;
      size_t allocs_in_next_buffer = n - (buffer_size - iid);
      RealType* ptr = &(*data_[bid])[iid];
      for (size_t i = 0; i < allocs_in_curr_buffer; ++i) {
        new (ptr + i) RealType();
      }
      allocate_buffer();
      bid = data_.size() - 1;
      iid = 0;
      total_size_ += n;

      RealType* ptr2 = &(*data_[bid])[iid];
      for (size_t i = 0; i < allocs_in_next_buffer; i++) {
        new (ptr2 + i) RealType();
      }
      return iterator(this, total_size_ - n, total_size_ - n, total_size_);
    }
  }

  /** @brief number of elements */
  size_t size() const { return total_size_; }

  /** @brief total capacity */
  size_t capacity() const { return data_.size() * buffer_size; }

  /** @brief iterator helpers */
  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, total_size_); }
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, total_size_); }

  iterator last_checkpoint()
  {
    return iterator(this, checkpoints_.back(), 0, total_size_);
  }
  iterator first_checkpoint()
  {
    return iterator(this, checkpoints_[0], 0, total_size_);
  };
  iterator checkpoint_at(size_t index)
  {
    return iterator(this, checkpoints_[index], 0, total_size_);
  };

  /** @brief searches for item in allocator
   *  only used to find gradient nodes for propagation */
  iterator find(const RealType* const item)
  {
    return std::find_if(
      begin(), end(), [&](const RealType& val) { return &val == item; });
  }
  /** @brief vector like access,
   *  currently unused anywhere but very useful for debugging
   */
  RealType& operator[](std::size_t i)
  {
    BOOST_MATH_ASSERT(i < total_size_);
    return (*data_[i / buffer_size])[i % buffer_size];
  }
  const RealType& operator[](std::size_t i) const
  {
    BOOST_MATH_ASSERT(i < total_size_);
    return (*data_[i / buffer_size])[i % buffer_size];
  }
};
} // namespace detail
} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost

#endif
