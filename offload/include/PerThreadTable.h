//===-- PerThreadTable.h -- PerThread Storage Structure ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Table indexed with one entry per thread.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PERTHREADTABLE_H
#define OFFLOAD_PERTHREADTABLE_H

#include <list>
#include <memory>
#include <mutex>
#include <type_traits>

template <typename ObjectType> struct PerThread {
  struct PerThreadData {
    std::unique_ptr<ObjectType> ThEntry;
  };

  std::mutex Mtx;
  std::list<std::shared_ptr<PerThreadData>> ThreadDataList;

  // define default constructors, disable copy and move constructors
  PerThread() = default;
  PerThread(const PerThread &) = delete;
  PerThread(PerThread &&) = delete;
  PerThread &operator=(const PerThread &) = delete;
  PerThread &operator=(PerThread &&) = delete;
  ~PerThread() {
    std::lock_guard<std::mutex> Lock(Mtx);
    ThreadDataList.clear();
  }

private:
  PerThreadData &getThreadData() {
    static thread_local std::shared_ptr<PerThreadData> ThData = nullptr;
    if (!ThData) {
      ThData = std::make_shared<PerThreadData>();
      std::lock_guard<std::mutex> Lock(Mtx);
      ThreadDataList.push_back(ThData);
    }
    return *ThData;
  }

protected:
  ObjectType &getThreadEntry() {
    auto &ThData = getThreadData();
    if (ThData.ThEntry)
      return *ThData.ThEntry;
    ThData.ThEntry = std::make_unique<ObjectType>();
    return *ThData.ThEntry;
  }

public:
  ObjectType &get() { return getThreadEntry(); }

  template <class F> void clear(F f) {
    std::lock_guard<std::mutex> Lock(Mtx);
    for (auto ThData : ThreadDataList) {
      if (!ThData->ThEntry)
        continue;
      f(*ThData->ThEntry);
    }
    ThreadDataList.clear();
  }
};

// Using an STL container (such as std::vector) indexed by thread ID has
// too many race conditions issues so we store each thread entry into a
// thread_local variable.
// T is the container type used to store the objects, e.g., std::vector,
// std::set, etc. by each thread. O is the type of the stored objects e.g.,
// omp_interop_val_t *, ...
template <typename ContainerType, typename ObjectType> struct PerThreadTable {
  using iterator = typename ContainerType::iterator;

  template <typename, typename = std::void_t<>>
  struct has_iterator : std::false_type {};
  template <typename T>
  struct has_iterator<T, std::void_t<typename T::iterator>> : std::true_type {};

  template <typename T, typename = std::void_t<>>
  struct has_clear : std::false_type {};
  template <typename T>
  struct has_clear<T, std::void_t<decltype(std::declval<T>().clear())>>
      : std::true_type {};

  template <typename T, typename = std::void_t<>>
  struct has_clearAll : std::false_type {};
  template <typename T>
  struct has_clearAll<T, std::void_t<decltype(std::declval<T>().clearAll(1))>>
      : std::true_type {};

  template <typename, typename = std::void_t<>>
  struct is_associative : std::false_type {};
  template <typename T>
  struct is_associative<T, std::void_t<typename T::mapped_type>>
      : std::true_type {};

  struct PerThreadData {
    size_t NElements = 0;
    std::unique_ptr<ContainerType> ThEntry;
  };

  std::mutex Mtx;
  std::list<std::shared_ptr<PerThreadData>> ThreadDataList;

  // define default constructors, disable copy and move constructors
  PerThreadTable() = default;
  PerThreadTable(const PerThreadTable &) = delete;
  PerThreadTable(PerThreadTable &&) = delete;
  PerThreadTable &operator=(const PerThreadTable &) = delete;
  PerThreadTable &operator=(PerThreadTable &&) = delete;
  ~PerThreadTable() {
    std::lock_guard<std::mutex> Lock(Mtx);
    ThreadDataList.clear();
  }

private:
  PerThreadData &getThreadData() {
    static thread_local std::shared_ptr<PerThreadData> ThData = nullptr;
    if (!ThData) {
      ThData = std::make_shared<PerThreadData>();
      std::lock_guard<std::mutex> Lock(Mtx);
      ThreadDataList.push_back(ThData);
    }
    return *ThData;
  }

protected:
  ContainerType &getThreadEntry() {
    auto &ThData = getThreadData();
    if (ThData.ThEntry)
      return *ThData.ThEntry;
    ThData.ThEntry = std::make_unique<ContainerType>();
    return *ThData.ThEntry;
  }

  size_t &getThreadNElements() {
    auto &ThData = getThreadData();
    return ThData.NElements;
  }

  void setNElements(size_t Size) {
    auto &NElements = getThreadNElements();
    NElements = Size;
  }

public:
  void add(ObjectType obj) {
    auto &Entry = getThreadEntry();
    auto &NElements = getThreadNElements();
    NElements++;
    Entry.add(obj);
  }

  iterator erase(iterator it) {
    auto &Entry = getThreadEntry();
    auto &NElements = getThreadNElements();
    NElements--;
    return Entry.erase(it);
  }

  size_t size() { return getThreadNElements(); }

  // Iterators to traverse objects owned by
  // the current thread
  iterator begin() {
    auto &Entry = getThreadEntry();
    return Entry.begin();
  }
  iterator end() {
    auto &Entry = getThreadEntry();
    return Entry.end();
  }

  template <class F> void clear(F f) {
    std::lock_guard<std::mutex> Lock(Mtx);
    for (auto ThData : ThreadDataList) {
      if (!ThData->ThEntry || ThData->NElements == 0)
        continue;
      if constexpr (has_clearAll<ContainerType>::value) {
        ThData->ThEntry->clearAll(f);
      } else if constexpr (has_iterator<ContainerType>::value &&
                           has_clear<ContainerType>::value) {
        for (auto &Obj : *ThData->ThEntry) {
          if constexpr (is_associative<ContainerType>::value) {
            f(Obj.second);
          } else {
            f(Obj);
          }
        }
        ThData->ThEntry->clear();
      } else {
        static_assert(true, "Container type not supported");
      }
      ThData->NElements = 0;
    }
    ThreadDataList.clear();
  }
};

template <typename T, typename = std::void_t<>> struct ContainerValueType {
  using type = typename T::value_type;
};
template <typename T>
struct ContainerValueType<T, std::void_t<typename T::mapped_type>> {
  using type = typename T::mapped_type;
};

template <typename ContainerType, size_t reserveSize = 0>
struct PerThreadContainer
    : public PerThreadTable<ContainerType,
                            typename ContainerValueType<ContainerType>::type> {

  // helpers
  template <typename T, typename = std::void_t<>> struct indexType {
    using type = typename T::size_type;
  };
  template <typename T> struct indexType<T, std::void_t<typename T::key_type>> {
    using type = typename T::key_type;
  };
  template <typename T, typename = std::void_t<>>
  struct has_resize : std::false_type {};
  template <typename T>
  struct has_resize<T, std::void_t<decltype(std::declval<T>().resize(1))>>
      : std::true_type {};

  template <typename T, typename = std::void_t<>>
  struct has_reserve : std::false_type {};
  template <typename T>
  struct has_reserve<T, std::void_t<decltype(std::declval<T>().reserve(1))>>
      : std::true_type {};

  using IndexType = typename indexType<ContainerType>::type;
  using ObjectType = typename ContainerValueType<ContainerType>::type;

  // Get the object for the given index in the current thread
  ObjectType &get(IndexType Index) {
    auto &Entry = this->getThreadEntry();

    // specialized code for vector-like containers
    if constexpr (has_resize<ContainerType>::value) {
      if (Index >= Entry.size()) {
        if constexpr (has_reserve<ContainerType>::value && reserveSize > 0) {
          if (Entry.capacity() < reserveSize)
            Entry.reserve(reserveSize);
        }
        // If the index is out of bounds, try resize the container
        Entry.resize(Index + 1);
      }
    }
    ObjectType &Ret = Entry[Index];
    this->setNElements(Entry.size());
    return Ret;
  }
};

#endif
